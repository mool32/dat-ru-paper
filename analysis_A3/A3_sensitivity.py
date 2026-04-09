"""
A3: Sensitivity analysis for user_agent identification quality
DAT-RU dataset
"""
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy import stats
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 150
plt.rcParams['figure.facecolor'] = 'white'

# ── Load data ──
df = pd.read_csv("/Users/teo/Downloads/DAT-RU Results - Sheet1 (1).csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

print(f"Total records: {len(df)}")
print(f"Unique UAs: {df['user_agent'].nunique()}")

# ══════════════════════════════════════════════════════════════════
# STEP 1: Parse user_agent strings
# ══════════════════════════════════════════════════════════════════

def parse_ua(ua):
    """Extract device model, OS version, browser version, in-app status."""
    result = {
        'device_model': None,
        'os': None,
        'os_version': None,
        'browser': None,
        'browser_version': None,
        'is_inapp': False,
        'inapp_source': None,
        'device_class': None,  # iPhone, Android, Desktop
        'specificity': 'generic',  # specific vs generic
    }

    if not isinstance(ua, str):
        return result

    # In-app browser detection
    if 'FB_IAB' in ua or 'FBAN/' in ua or 'FBIOS' in ua:
        result['is_inapp'] = True
        result['inapp_source'] = 'Facebook'
    elif 'Instagram' in ua:
        result['is_inapp'] = True
        result['inapp_source'] = 'Instagram'
    elif 'Telegram' in ua or 'TelegramBot' in ua:
        result['is_inapp'] = True
        result['inapp_source'] = 'Telegram'
    elif 'DuckDuckGo' in ua:
        result['is_inapp'] = True
        result['inapp_source'] = 'DuckDuckGo'
    elif '; wv)' in ua:
        result['is_inapp'] = True
        result['inapp_source'] = 'WebView'

    # ── iPhone ──
    if 'iPhone' in ua:
        result['device_class'] = 'iPhone'
        # Get iOS version
        m = re.search(r'CPU iPhone OS (\d+[_\.]\d+(?:[_\.]\d+)?)', ua)
        if m:
            result['os'] = 'iOS'
            result['os_version'] = m.group(1).replace('_', '.')

        # Get iPhone model from FB tags
        m = re.search(r'FBDV/(iPhone[\d,]+)', ua)
        if m:
            result['device_model'] = m.group(1)
            result['specificity'] = 'specific'
        else:
            # Try to get from build number or other markers
            # Without FBDV, iPhone UAs are somewhat generic (only OS version differentiates)
            result['device_model'] = 'iPhone_unknown'
            # iPhone + specific iOS version + browser version is moderately specific
            result['specificity'] = 'moderate'

        # Browser
        if 'CriOS' in ua:
            m = re.search(r'CriOS/([\d.]+)', ua)
            if m: result['browser'] = 'Chrome'; result['browser_version'] = m.group(1)
        elif 'FxiOS' in ua:
            m = re.search(r'FxiOS/([\d.]+)', ua)
            if m: result['browser'] = 'Firefox'; result['browser_version'] = m.group(1)
        elif 'Safari' in ua:
            m = re.search(r'Version/([\d.]+)', ua)
            if m: result['browser'] = 'Safari'; result['browser_version'] = m.group(1)
            else: result['browser'] = 'Safari'

        return result

    # ── iPad ──
    if 'iPad' in ua:
        result['device_class'] = 'iPad'
        m = re.search(r'CPU OS (\d+[_\.]\d+(?:[_\.]\d+)?)', ua)
        if m:
            result['os'] = 'iPadOS'
            result['os_version'] = m.group(1).replace('_', '.')
        m = re.search(r'FBDV/(iPad[\d,]+)', ua)
        if m:
            result['device_model'] = m.group(1)
            result['specificity'] = 'specific'
        else:
            result['device_model'] = 'iPad_unknown'
            result['specificity'] = 'moderate'
        return result

    # ── Android ──
    if 'Android' in ua:
        result['device_class'] = 'Android'
        m = re.search(r'Android (\d+[\d.]*)', ua)
        if m:
            result['os'] = 'Android'
            result['os_version'] = m.group(1)

        # Device model: between "Android XX; " and " Build/"
        m = re.search(r'Android [\d.]+;\s*(.+?)\s*(?:Build/|[;\)])', ua)
        if m:
            model = m.group(1).strip()
            # Check if it's a specific model or generic
            generic_patterns = [r'^K$', r'^moto ', r'^SM-', r'^Redmi', r'^Pixel',
                               r'^SAMSUNG', r'^M\d{4}', r'^V\d{4}', r'^RMX']
            result['device_model'] = model
            # A single letter like "K" is super generic
            if len(model) <= 2 or model in ('K', 'M', 'U'):
                result['specificity'] = 'generic'
            else:
                result['specificity'] = 'specific'
        else:
            result['device_model'] = 'Android_unknown'
            result['specificity'] = 'generic'

        # Chrome version
        m = re.search(r'Chrome/([\d.]+)', ua)
        if m: result['browser'] = 'Chrome'; result['browser_version'] = m.group(1)

        return result

    # ── Desktop / Other ──
    result['device_class'] = 'Desktop'

    if 'Windows' in ua:
        result['os'] = 'Windows'
        m = re.search(r'Windows NT ([\d.]+)', ua)
        if m: result['os_version'] = m.group(1)
    elif 'Macintosh' in ua or 'Mac OS X' in ua:
        result['os'] = 'macOS'
        m = re.search(r'Mac OS X ([\d_]+)', ua)
        if m: result['os_version'] = m.group(1).replace('_', '.')
    elif 'Linux' in ua:
        result['os'] = 'Linux'
    elif 'CrOS' in ua:
        result['os'] = 'ChromeOS'

    # Browser
    if 'Edg/' in ua:
        m = re.search(r'Edg/([\d.]+)', ua)
        if m: result['browser'] = 'Edge'; result['browser_version'] = m.group(1)
    elif 'OPR/' in ua:
        m = re.search(r'OPR/([\d.]+)', ua)
        if m: result['browser'] = 'Opera'; result['browser_version'] = m.group(1)
    elif 'YaBrowser/' in ua:
        m = re.search(r'YaBrowser/([\d.]+)', ua)
        if m: result['browser'] = 'Yandex'; result['browser_version'] = m.group(1)
    elif 'Firefox/' in ua:
        m = re.search(r'Firefox/([\d.]+)', ua)
        if m: result['browser'] = 'Firefox'; result['browser_version'] = m.group(1)
    elif 'Chrome/' in ua:
        m = re.search(r'Chrome/([\d.]+)', ua)
        if m: result['browser'] = 'Chrome'; result['browser_version'] = m.group(1)
    elif 'Safari/' in ua:
        result['browser'] = 'Safari'
        m = re.search(r'Version/([\d.]+)', ua)
        if m: result['browser_version'] = m.group(1)

    # Desktop UAs are very generic — many people share same OS+browser
    result['specificity'] = 'generic'
    result['device_model'] = f"Desktop_{result['os']}_{result['browser']}"

    return result

# Parse all UAs
print("\nParsing user agents...")
ua_parsed = df['user_agent'].apply(parse_ua).apply(pd.Series)
df = pd.concat([df, ua_parsed], axis=1)

print(f"\nDevice class distribution:")
print(df['device_class'].value_counts())
print(f"\nSpecificity distribution:")
print(df['specificity'].value_counts())
print(f"\nIn-app browser distribution:")
print(df['is_inapp'].value_counts())

# ══════════════════════════════════════════════════════════════════
# STEP 2: Collision probability analysis
# ══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("STEP 2: Collision Probability Analysis")
print("="*70)

# Group by UA to get per-user stats
ua_groups = df.groupby('user_agent').agg(
    n_attempts=('score', 'count'),
    specificity=('specificity', 'first'),
    device_class=('device_class', 'first'),
    device_model=('device_model', 'first'),
    is_inapp=('is_inapp', 'first'),
    os=('os', 'first'),
    os_version=('os_version', 'first'),
    browser=('browser', 'first'),
    browser_version=('browser_version', 'first'),
).reset_index()

print(f"\nTotal unique UAs: {len(ua_groups)}")
print(f"\nSpecificity by UA count:")
spec_counts = ua_groups['specificity'].value_counts()
print(spec_counts)
print(f"\nSpecificity by record count:")
for s in ['specific', 'moderate', 'generic']:
    mask = ua_groups['specificity'] == s
    n_uas = mask.sum()
    n_records = ua_groups.loc[mask, 'n_attempts'].sum()
    print(f"  {s}: {n_uas} UAs, {n_records} records ({100*n_records/len(df):.1f}%)")

# For specific UAs: how unique are they?
specific_uas = ua_groups[ua_groups['specificity'] == 'specific']
print(f"\n--- Specific UAs ---")
print(f"Device models in 'specific' category:")
model_counts = specific_uas['device_model'].value_counts()
print(model_counts.head(20))

# For generic UAs: estimate collision probability
generic_uas = ua_groups[ua_groups['specificity'] == 'generic']
print(f"\n--- Generic UAs ---")
print(f"Top generic device models:")
print(generic_uas['device_model'].value_counts().head(15))

# Desktop breakdown
desktop_uas = ua_groups[ua_groups['device_class'] == 'Desktop']
print(f"\n--- Desktop UAs ({len(desktop_uas)} unique) ---")
print(f"OS distribution:")
print(desktop_uas['os'].value_counts())
print(f"Browser distribution:")
print(desktop_uas['browser'].value_counts())

# ── Detailed specificity refinement ──
# For "specific" Android models, check if the FULL UA string is unique enough
# Even with same model, different OS+browser versions help
print(f"\n--- UA string uniqueness ---")
# Count how many records each UA has
ua_attempt_dist = ua_groups['n_attempts'].describe()
print(f"Attempts per UA:\n{ua_attempt_dist}")

# Heavy users (>=10 attempts)
heavy = ua_groups[ua_groups['n_attempts'] >= 10]
print(f"\nHeavy users (>=10 attempts): {len(heavy)} UAs, {heavy['n_attempts'].sum()} records")
print(f"  Specific: {(heavy['specificity']=='specific').sum()}")
print(f"  Moderate: {(heavy['specificity']=='moderate').sum()}")
print(f"  Generic: {(heavy['specificity']=='generic').sum()}")

# ══════════════════════════════════════════════════════════════════
# STEP 2b: Create a combined specificity score
# ══════════════════════════════════════════════════════════════════

# Upgrade 'moderate' iPhones with very specific iOS versions + in-app browser
# (FB in-app adds FBDV which makes them 'specific' already)
# For remaining 'moderate': iPhone + specific iOS minor version + Safari version
# = reasonably specific for test-retest (not perfect, but better than desktop)

# For analysis purposes, define "high confidence" = specific + moderate
# and "low confidence" = generic

df['confidence'] = df['specificity'].map({
    'specific': 'high',
    'moderate': 'medium',
    'generic': 'low'
})

ua_groups['confidence'] = ua_groups['specificity'].map({
    'specific': 'high',
    'moderate': 'medium',
    'generic': 'low'
})

# ══════════════════════════════════════════════════════════════════
# STEP 3: Sensitivity analysis — repeat B1 on specific UAs
# ══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("STEP 3: Sensitivity Analysis — B1 metrics by confidence level")
print("="*70)

def compute_test_retest(data, label=""):
    """Compute test-retest r for attempt 1 vs attempt 2."""
    # Number attempts per UA
    data = data.sort_values(['user_agent', 'timestamp'])
    data['attempt'] = data.groupby('user_agent').cumcount() + 1

    # Get users with at least 2 attempts
    users_2plus = data[data['attempt'] >= 2]['user_agent'].unique()

    a1 = data[(data['user_agent'].isin(users_2plus)) & (data['attempt'] == 1)].set_index('user_agent')['score']
    a2 = data[(data['user_agent'].isin(users_2plus)) & (data['attempt'] == 2)].set_index('user_agent')['score']

    common = a1.index.intersection(a2.index)
    if len(common) < 10:
        return None, None, len(common)

    r, p = stats.pearsonr(a1[common], a2[common])
    print(f"  {label}: test-retest r={r:.3f} (p={p:.2e}), N={len(common)} users")
    return r, p, len(common)

def compute_practice_slope(data, min_attempts=5, label=""):
    """Compute practice effect slope using mixed-effects proxy (OLS on attempt)."""
    data = data.sort_values(['user_agent', 'timestamp'])
    data['attempt'] = data.groupby('user_agent').cumcount() + 1

    # Users with >= min_attempts
    ua_counts = data.groupby('user_agent').size()
    eligible = ua_counts[ua_counts >= min_attempts].index
    subset = data[data['user_agent'].isin(eligible)].copy()

    if len(subset) < 20:
        return None, None, 0

    # Simple approach: within-user slopes
    slopes = []
    for ua, grp in subset.groupby('user_agent'):
        if len(grp) >= min_attempts:
            s, _, _, _, _ = stats.linregress(grp['attempt'], grp['score'])
            slopes.append(s)

    slopes = np.array(slopes)
    mean_slope = np.mean(slopes)
    t_stat, p_val = stats.ttest_1samp(slopes, 0)
    print(f"  {label}: mean slope={mean_slope:.4f}, t={t_stat:.2f}, p={p_val:.3f}, N={len(slopes)} users")
    return mean_slope, p_val, len(slopes)

# Full sample
print("\n--- Full sample ---")
r_full, p_full, n_full = compute_test_retest(df.copy(), "Full")
slope_full, sp_full, ns_full = compute_practice_slope(df.copy(), label="Full")

# By confidence level
results = {}
for conf in ['high', 'medium', 'low']:
    subset = df[df['confidence'] == conf].copy()
    print(f"\n--- Confidence: {conf} ({subset['user_agent'].nunique()} UAs, {len(subset)} records) ---")
    r, p, n = compute_test_retest(subset, conf)
    s, sp, ns = compute_practice_slope(subset, label=conf)
    results[conf] = {'r': r, 'r_p': p, 'r_n': n, 'slope': s, 'slope_p': sp, 'slope_n': ns}

# High + Medium combined (= all non-generic)
subset_hm = df[df['confidence'].isin(['high', 'medium'])].copy()
print(f"\n--- High + Medium combined ({subset_hm['user_agent'].nunique()} UAs, {len(subset_hm)} records) ---")
r_hm, p_hm, n_hm = compute_test_retest(subset_hm, "High+Med")
s_hm, sp_hm, ns_hm = compute_practice_slope(subset_hm, label="High+Med")

# ══════════════════════════════════════════════════════════════════
# STEP 4: Time-pattern analysis for heavy users
# ══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("STEP 4: Time-pattern analysis for heavy users")
print("="*70)

heavy_threshold = 50
heavy_uas = df.groupby('user_agent').filter(lambda x: len(x) >= heavy_threshold)
heavy_ua_list = heavy_uas['user_agent'].unique()
print(f"UAs with >= {heavy_threshold} attempts: {len(heavy_ua_list)}")

# For each heavy UA, analyze time patterns
suspicious_uas = []
genuine_uas = []

for ua in heavy_ua_list:
    grp = df[df['user_agent'] == ua].copy()
    grp['hour'] = grp['timestamp'].dt.hour
    grp['date'] = grp['timestamp'].dt.date

    n = len(grp)
    n_days = grp['date'].nunique()
    hours_active = grp['hour'].unique()

    # Suspicion criteria:
    # 1. Active across very wide hour range (e.g., both 2-5am AND 10am-5pm regularly)
    # 2. Multiple submissions in impossibly short time from "same device"
    # 3. Activity spanning too many days with uniform distribution (bot-like)

    # Check for bimodal time distribution (night + day)
    night_count = grp[grp['hour'].isin(range(1, 6))].shape[0]  # 1am-5am
    day_count = grp[grp['hour'].isin(range(9, 18))].shape[0]   # 9am-5pm

    night_frac = night_count / n if n > 0 else 0
    day_frac = day_count / n if n > 0 else 0

    # Bimodal: substantial activity in both night and day
    bimodal = (night_frac > 0.1 and day_frac > 0.3)

    # Check inter-submission intervals
    if len(grp) > 1:
        intervals = grp['timestamp'].diff().dt.total_seconds().dropna()
        very_fast = (intervals < 30).sum()  # Less than 30 seconds between submissions
    else:
        very_fast = 0

    # Check hour entropy (uniform = suspicious for a single person)
    hour_counts = grp['hour'].value_counts(normalize=True)
    hour_entropy = stats.entropy(hour_counts)
    max_entropy = np.log(24)  # uniform across 24 hours
    entropy_ratio = hour_entropy / max_entropy

    info = {
        'ua': ua[:80] + '...' if len(ua) > 80 else ua,
        'n': n,
        'n_days': n_days,
        'n_hours_active': len(hours_active),
        'night_frac': night_frac,
        'day_frac': day_frac,
        'bimodal': bimodal,
        'very_fast_submissions': very_fast,
        'hour_entropy_ratio': entropy_ratio,
        'specificity': grp['specificity'].iloc[0],
        'device_class': grp['device_class'].iloc[0],
    }

    # Flag as suspicious if bimodal OR very high entropy OR many fast submissions
    is_suspicious = bimodal or entropy_ratio > 0.85 or very_fast > 3
    info['suspicious'] = is_suspicious

    if is_suspicious:
        suspicious_uas.append(info)
    else:
        genuine_uas.append(info)

print(f"\nGenuine-looking heavy users: {len(genuine_uas)}")
print(f"Suspicious heavy users (likely collisions): {len(suspicious_uas)}")

if suspicious_uas:
    print(f"\nSuspicious users details:")
    for s in sorted(suspicious_uas, key=lambda x: -x['n'])[:10]:
        print(f"  N={s['n']}, days={s['n_days']}, hours={s['n_hours_active']}, "
              f"night={s['night_frac']:.2f}, entropy={s['hour_entropy_ratio']:.2f}, "
              f"fast={s['very_fast_submissions']}, {s['specificity']}, "
              f"bimodal={s['bimodal']}")

print(f"\nGenuine heavy users by specificity:")
gen_spec = Counter(g['specificity'] for g in genuine_uas)
print(f"  {dict(gen_spec)}")

susp_spec = Counter(s['specificity'] for s in suspicious_uas)
print(f"Suspicious by specificity: {dict(susp_spec)}")

# ══════════════════════════════════════════════════════════════════
# STEP 5: Summary statistics
# ══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("STEP 5: Summary")
print("="*70)

total_records = len(df)
total_uas = df['user_agent'].nunique()

# Records by confidence
for conf in ['high', 'medium', 'low']:
    mask = df['confidence'] == conf
    n_rec = mask.sum()
    n_ua = df.loc[mask, 'user_agent'].nunique()
    print(f"  {conf}: {n_ua} UAs ({100*n_ua/total_uas:.1f}%), {n_rec} records ({100*n_rec/total_records:.1f}%)")

# Collision-free estimate
# For "specific" UAs (model+OS+browser fully identified): collision probability low
# For "moderate" UAs (iPhone + iOS version): moderate collision risk
# For "generic" (desktop, generic Android): high collision risk

# Among heavy users, what fraction are suspicious?
if heavy_ua_list.size > 0:
    susp_frac = len(suspicious_uas) / len(heavy_ua_list)
    print(f"\nAmong heavy users (>={heavy_threshold} attempts):")
    print(f"  Suspicious (likely collisions): {len(suspicious_uas)}/{len(heavy_ua_list)} ({100*susp_frac:.1f}%)")

    susp_records = sum(s['n'] for s in suspicious_uas)
    gen_records = sum(g['n'] for g in genuine_uas)
    print(f"  Records in suspicious: {susp_records}")
    print(f"  Records in genuine: {gen_records}")

print(f"\n--- Key comparison ---")
print(f"  Test-retest r (full sample):        {r_full:.3f} (N={n_full})")
if results['high']['r'] is not None:
    print(f"  Test-retest r (high confidence):    {results['high']['r']:.3f} (N={results['high']['r_n']})")
if results['medium']['r'] is not None:
    print(f"  Test-retest r (medium confidence):  {results['medium']['r']:.3f} (N={results['medium']['r_n']})")
if results['low']['r'] is not None:
    print(f"  Test-retest r (low confidence):     {results['low']['r']:.3f} (N={results['low']['r_n']})")
if r_hm is not None:
    print(f"  Test-retest r (high+medium):        {r_hm:.3f} (N={n_hm})")

print(f"\n  Practice slope (full):     {slope_full:.4f} (p={sp_full:.3f})")
for conf in ['high', 'medium', 'low']:
    if results[conf]['slope'] is not None:
        print(f"  Practice slope ({conf:6s}):   {results[conf]['slope']:.4f} (p={results[conf]['slope_p']:.3f})")

# ══════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════

# ── Plot 1: UA composition ──
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Device class pie
device_counts = df['device_class'].value_counts()
axes[0].pie(device_counts.values, labels=device_counts.index, autopct='%1.1f%%',
           colors=['#4C72B0', '#55A868', '#C44E52', '#8172B2'])
axes[0].set_title('Device Class (by records)')

# Specificity pie
spec_counts_rec = df['specificity'].value_counts()
colors_spec = {'specific': '#2ecc71', 'moderate': '#f39c12', 'generic': '#e74c3c'}
axes[1].pie(spec_counts_rec.values, labels=spec_counts_rec.index, autopct='%1.1f%%',
           colors=[colors_spec.get(x, '#999') for x in spec_counts_rec.index])
axes[1].set_title('UA Specificity (by records)')

# In-app pie
inapp_counts = df['is_inapp'].value_counts()
axes[2].pie(inapp_counts.values, labels=['Regular browser', 'In-app browser'], autopct='%1.1f%%',
           colors=['#3498db', '#e67e22'])
axes[2].set_title('In-app Browser (by records)')

plt.tight_layout()
plt.savefig('/Users/teo/Downloads/datcreativity/analysis_A3/A3_ua_composition.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Plot 2: Test-retest comparison ──
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Bar chart of test-retest r
categories = ['Full', 'High\n(specific)', 'Medium\n(moderate)', 'Low\n(generic)', 'High+Med']
r_values = [r_full,
            results['high']['r'] if results['high']['r'] else 0,
            results['medium']['r'] if results['medium']['r'] else 0,
            results['low']['r'] if results['low']['r'] else 0,
            r_hm if r_hm else 0]
n_values = [n_full,
            results['high']['r_n'],
            results['medium']['r_n'],
            results['low']['r_n'],
            n_hm]

colors_bar = ['#34495e', '#2ecc71', '#f39c12', '#e74c3c', '#3498db']
bars = axes[0].bar(categories, r_values, color=colors_bar, edgecolor='black', linewidth=0.5)
for bar, n in zip(bars, n_values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'N={n}', ha='center', va='bottom', fontsize=8)
axes[0].set_ylabel('Pearson r')
axes[0].set_title('Test-Retest Reliability by UA Confidence')
axes[0].set_ylim(0, max(r_values) * 1.3 if max(r_values) > 0 else 0.5)
axes[0].axhline(y=r_full, color='gray', linestyle='--', alpha=0.5, label=f'Full sample r={r_full:.3f}')
axes[0].legend(fontsize=8)

# Bar chart of practice slopes
slope_values = [slope_full,
                results['high']['slope'] if results['high']['slope'] else 0,
                results['medium']['slope'] if results['medium']['slope'] else 0,
                results['low']['slope'] if results['low']['slope'] else 0,
                s_hm if s_hm else 0]
ns_values = [ns_full,
             results['high']['slope_n'],
             results['medium']['slope_n'],
             results['low']['slope_n'],
             ns_hm]

bars2 = axes[1].bar(categories, slope_values, color=colors_bar, edgecolor='black', linewidth=0.5)
for bar, n in zip(bars2, ns_values):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.0005 if bar.get_height() >= 0 else bar.get_height() - 0.001,
                f'N={n}', ha='center', va='bottom' if bar.get_height() >= 0 else 'top', fontsize=8)
axes[1].set_ylabel('Mean within-user slope')
axes[1].set_title('Practice Effect Slope by UA Confidence')
axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.savefig('/Users/teo/Downloads/datcreativity/analysis_A3/A3_sensitivity_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Plot 3: Heavy user time patterns ──
# Plot hourly distributions for a few suspicious vs genuine heavy users
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Top 3 suspicious
for i, s in enumerate(sorted(suspicious_uas, key=lambda x: -x['n'])[:3]):
    ua_full = df[df['user_agent'].str.startswith(s['ua'][:40])]['user_agent'].iloc[0]
    grp = df[df['user_agent'] == ua_full]
    hours = grp['timestamp'].dt.hour
    axes[0, i].hist(hours, bins=24, range=(0, 24), color='#e74c3c', edgecolor='black', alpha=0.7)
    axes[0, i].set_title(f"SUSPICIOUS (N={s['n']})\nentropy={s['hour_entropy_ratio']:.2f}", fontsize=9)
    axes[0, i].set_xlabel('Hour of day')
    axes[0, i].set_ylabel('Count')
    axes[0, i].set_xlim(0, 24)

# Top 3 genuine
for i, g in enumerate(sorted(genuine_uas, key=lambda x: -x['n'])[:3]):
    ua_full = df[df['user_agent'].str.startswith(g['ua'][:40])]['user_agent'].iloc[0]
    grp = df[df['user_agent'] == ua_full]
    hours = grp['timestamp'].dt.hour
    axes[1, i].hist(hours, bins=24, range=(0, 24), color='#2ecc71', edgecolor='black', alpha=0.7)
    axes[1, i].set_title(f"GENUINE (N={g['n']})\nentropy={g['hour_entropy_ratio']:.2f}", fontsize=9)
    axes[1, i].set_xlabel('Hour of day')
    axes[1, i].set_ylabel('Count')
    axes[1, i].set_xlim(0, 24)

plt.suptitle('Hourly Activity Patterns: Suspicious vs Genuine Heavy Users', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('/Users/teo/Downloads/datcreativity/analysis_A3/A3_heavy_user_patterns.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Plot 4: Collision risk summary ──
fig, ax = plt.subplots(figsize=(10, 6))

# Create a stacked bar showing record counts by confidence and device class
pivot = df.groupby(['device_class', 'confidence']).size().unstack(fill_value=0)
pivot = pivot[['high', 'medium', 'low']]  # order columns
pivot.plot(kind='barh', stacked=True, ax=ax,
           color=['#2ecc71', '#f39c12', '#e74c3c'],
           edgecolor='black', linewidth=0.5)
ax.set_xlabel('Number of Records')
ax.set_title('Records by Device Class and Identification Confidence')
ax.legend(title='Confidence', loc='lower right')

for container in ax.containers:
    labels = [f'{int(v.get_width())}' if v.get_width() > 200 else '' for v in container]
    ax.bar_label(container, labels=labels, label_type='center', fontsize=8)

plt.tight_layout()
plt.savefig('/Users/teo/Downloads/datcreativity/analysis_A3/A3_collision_risk.png', dpi=150, bbox_inches='tight')
plt.close()

# ── Plot 5: Test-retest scatterplots side by side ──
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, (conf_label, conf_filter) in enumerate([
    ('High Confidence (specific)', ['high']),
    ('Low Confidence (generic)', ['low']),
    ('Full Sample', ['high', 'medium', 'low'])
]):
    subset = df[df['confidence'].isin(conf_filter)].copy()
    subset = subset.sort_values(['user_agent', 'timestamp'])
    subset['attempt'] = subset.groupby('user_agent').cumcount() + 1

    users_2plus = subset[subset['attempt'] >= 2]['user_agent'].unique()
    a1 = subset[(subset['user_agent'].isin(users_2plus)) & (subset['attempt'] == 1)].set_index('user_agent')['score']
    a2 = subset[(subset['user_agent'].isin(users_2plus)) & (subset['attempt'] == 2)].set_index('user_agent')['score']
    common = a1.index.intersection(a2.index)

    if len(common) >= 10:
        r_val, _ = stats.pearsonr(a1[common], a2[common])
        axes[idx].scatter(a1[common], a2[common], alpha=0.3, s=10, color=colors_bar[idx+1] if idx < 2 else '#34495e')
        axes[idx].set_xlabel('Attempt 1 score')
        axes[idx].set_ylabel('Attempt 2 score')
        axes[idx].set_title(f'{conf_label}\nr={r_val:.3f}, N={len(common)}')

        # Add identity line
        lims = [min(axes[idx].get_xlim()[0], axes[idx].get_ylim()[0]),
                max(axes[idx].get_xlim()[1], axes[idx].get_ylim()[1])]
        axes[idx].plot(lims, lims, 'k--', alpha=0.3)
        axes[idx].set_xlim(lims)
        axes[idx].set_ylim(lims)

plt.suptitle('Test-Retest Scatterplots by UA Confidence Level', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('/Users/teo/Downloads/datcreativity/analysis_A3/A3_test_retest_scatter.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nAll plots saved to /Users/teo/Downloads/datcreativity/analysis_A3/")
print("Done!")
