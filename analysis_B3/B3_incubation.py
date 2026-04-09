"""
B3: Incubation effect — does a break between attempts improve scores?
"""
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess
import warnings
warnings.filterwarnings('ignore')

OUT = '/Users/teo/Downloads/datcreativity/analysis_B3'

# ── 1. Load & classify ──────────────────────────────────────────────
df = pd.read_csv('/Users/teo/Downloads/DAT-RU Results - Sheet1 (1).csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

import re

def classify_ua(ua):
    if pd.isna(ua):
        return 'generic', None
    ua = str(ua)
    # iPhone specific (FBDV/iPhoneXX,Y or direct)
    m = re.search(r'(?:FBDV/|; )(iPhone\d+,\d+)', ua)
    if m:
        return 'specific', m.group(1)
    # Samsung
    m = re.search(r'(SM-[A-Z0-9]+)', ua)
    if m:
        return 'specific', m.group(1)
    # Pixel
    m = re.search(r'(Pixel\s*\d+[^;)]*)', ua)
    if m:
        return 'specific', m.group(1).strip()
    # Xiaomi / Redmi / POCO
    m = re.search(r'((?:Redmi|POCO|Mi)\s+[^;)]+)', ua)
    if m:
        return 'specific', m.group(1).strip()
    # Other Android models with specific build
    m = re.search(r'Android\s+\d+;\s+([A-Z]{2,}[\w\-]+)', ua)
    if m and m.group(1) != 'K':
        return 'specific', m.group(1)
    return 'generic', None

df[['ua_type', 'device_id']] = df['user_agent'].apply(lambda x: pd.Series(classify_ua(x)))

# Use full UA for specific users as identity proxy
df['user_key'] = np.where(df['ua_type'] == 'specific', df['user_agent'], None)

# Filter to specific-UA users with ≥5 attempts
specific = df[df['ua_type'] == 'specific'].copy()
user_counts = specific['user_key'].value_counts()
valid_users = user_counts[user_counts >= 5].index
print(f"Total records: {len(df)}")
print(f"Specific UA records: {len(specific)} ({100*len(specific)/len(df):.1f}%)")
print(f"Users (specific) with ≥5 attempts: {len(valid_users)}")

data = specific[specific['user_key'].isin(valid_users)].copy()
data = data.sort_values(['user_key', 'timestamp']).reset_index(drop=True)
print(f"Records from valid users: {len(data)}")
print(f"Attempts per user: median={user_counts[valid_users].median():.0f}, "
      f"mean={user_counts[valid_users].mean():.1f}, max={user_counts[valid_users].max()}")

# ── 2. Compute intervals & Δscore ──────────────────────────────────
pairs = []
for uid, grp in data.groupby('user_key'):
    grp = grp.sort_values('timestamp').reset_index(drop=True)
    for i in range(len(grp) - 1):
        interval_min = (grp.loc[i+1, 'timestamp'] - grp.loc[i, 'timestamp']).total_seconds() / 60
        dscore = grp.loc[i+1, 'score'] - grp.loc[i, 'score']
        pairs.append({
            'user': uid[:60],
            'attempt_n': i + 1,
            'attempt_n1': i + 2,
            'score_n': grp.loc[i, 'score'],
            'score_n1': grp.loc[i+1, 'score'],
            'dscore': dscore,
            'interval_min': interval_min,
            'timestamp_n': grp.loc[i, 'timestamp'],
            'timestamp_n1': grp.loc[i+1, 'timestamp'],
        })

pairs_df = pd.DataFrame(pairs)
pairs_df['log_interval'] = np.log10(pairs_df['interval_min'].clip(lower=0.1))
print(f"\nConsecutive pairs: {len(pairs_df)}")
print(f"Interval range: {pairs_df['interval_min'].min():.1f} - {pairs_df['interval_min'].max():.0f} minutes")

# ── 3. Define sessions ─────────────────────────────────────────────
# Within session: <10 min gap. New session: ≥360 min (6 hours).
data = data.copy()
data['prev_time'] = data.groupby('user_key')['timestamp'].shift(1)
data['gap_min'] = (data['timestamp'] - data['prev_time']).dt.total_seconds() / 60

# Assign session IDs
data['new_session'] = (data['gap_min'].isna()) | (data['gap_min'] >= 360)
data['session_id'] = data.groupby('user_key')['new_session'].cumsum()
data['session_key'] = data['user_key'].astype(str) + '_s' + data['session_id'].astype(str)

# Attempt within session
data['within_session_attempt'] = data.groupby('session_key').cumcount() + 1

# Classify pairs
pairs_df['transition_type'] = pd.cut(
    pairs_df['interval_min'],
    bins=[0, 10, 360, np.inf],
    labels=['within_session', 'short_break', 'between_session']
)

# More granular
def classify_interval(m):
    if m < 10: return 'no_break'
    elif m < 60: return 'short_break_10_60m'
    elif m < 360: return 'medium_break_1_6h'
    elif m < 960: return 'overnight_6_16h'
    else: return 'long_break_16h+'

pairs_df['interval_cat'] = pairs_df['interval_min'].apply(classify_interval)

# ── 4a. Scatter: log(interval) vs Δscore with LOWESS ───────────────
fig, ax = plt.subplots(figsize=(10, 6))
hb = ax.hexbin(pairs_df['log_interval'], pairs_df['dscore'],
               gridsize=40, cmap='YlOrRd', mincnt=1)
plt.colorbar(hb, ax=ax, label='Count')

# LOWESS
mask = np.isfinite(pairs_df['log_interval']) & np.isfinite(pairs_df['dscore'])
lw = lowess(pairs_df.loc[mask, 'dscore'], pairs_df.loc[mask, 'log_interval'], frac=0.3)
ax.plot(lw[:, 0], lw[:, 1], 'cyan', lw=3, label='LOWESS smooth')

ax.axhline(0, color='white', ls='--', alpha=0.7)
ax.set_xlabel('log₁₀(interval in minutes)')
ax.set_ylabel('Δscore (next − current)')
ax.set_title('B3: Score Change vs Inter-Attempt Interval')
# Add reference lines for time categories
for val, lbl in [(np.log10(10), '10 min'), (np.log10(360), '6 hours'),
                 (np.log10(960), '16 hours')]:
    ax.axvline(val, color='white', ls=':', alpha=0.5)
    ax.text(val, ax.get_ylim()[1]*0.95, lbl, color='white', fontsize=8, ha='center')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUT}/B3a_interval_vs_dscore_hex.png', dpi=150)
plt.close()
print("\n[4a] Hex plot saved.")

# ── 4b. Binary comparison: within vs between session ────────────────
within = pairs_df[pairs_df['transition_type'] == 'within_session']['dscore'].dropna()
between = pairs_df[pairs_df['transition_type'] == 'between_session']['dscore'].dropna()

t_stat, p_val = stats.ttest_ind(within, between, equal_var=False)
print(f"\n[4b] Within-session Δscore: mean={within.mean():.3f}, sd={within.std():.3f}, n={len(within)}")
print(f"     Between-session Δscore: mean={between.mean():.3f}, sd={between.std():.3f}, n={len(between)}")
print(f"     Welch t={t_stat:.3f}, p={p_val:.4f}")

# Cohen's d
pooled_sd = np.sqrt((within.std()**2 + between.std()**2) / 2)
cohens_d = (between.mean() - within.mean()) / pooled_sd
print(f"     Cohen's d (between - within) = {cohens_d:.3f}")

fig, ax = plt.subplots(figsize=(8, 5))
bp_data = [within.values, between.values]
bp = ax.boxplot(bp_data, labels=['Within session\n(<10 min)', 'Between session\n(>6 hours)'],
                patch_artist=True, showmeans=True, meanline=True)
bp['boxes'][0].set_facecolor('#4ECDC4')
bp['boxes'][1].set_facecolor('#FF6B6B')
ax.axhline(0, color='gray', ls='--')
ax.set_ylabel('Δscore')
ax.set_title(f'B3b: Within vs Between Session Score Change\n'
             f'Welch t={t_stat:.2f}, p={p_val:.4f}, d={cohens_d:.3f}')
plt.tight_layout()
plt.savefig(f'{OUT}/B3b_within_vs_between.png', dpi=150)
plt.close()

# ── 4c. Regression: Δscore ~ log(interval) + attempt_number ────────
reg_df = pairs_df[['dscore', 'log_interval', 'attempt_n']].dropna()
X = sm.add_constant(reg_df[['log_interval', 'attempt_n']])
model = sm.OLS(reg_df['dscore'], X).fit()
print(f"\n[4c] OLS: Δscore ~ log(interval) + attempt_n")
print(model.summary2().tables[1].to_string())

# Also partial correlation
from numpy.linalg import lstsq

def partial_corr(df, x, y, covars):
    """Partial correlation of x and y controlling for covars."""
    # Residualize x
    X_cov = sm.add_constant(df[covars])
    res_x = df[x] - sm.OLS(df[x], X_cov).fit().predict(X_cov)
    res_y = df[y] - sm.OLS(df[y], X_cov).fit().predict(X_cov)
    return stats.pearsonr(res_x, res_y)

r_partial, p_partial = partial_corr(reg_df, 'log_interval', 'dscore', ['attempt_n'])
print(f"     Partial corr(log_interval, Δscore | attempt_n): r={r_partial:.4f}, p={p_partial:.4f}")

# ── 4d. Classic incubation: first of new session vs last of previous ─
session_edges = []
for uid, udata in data.groupby('user_key'):
    sessions = udata.groupby('session_key')
    skeys = sorted(sessions.groups.keys())
    for i in range(len(skeys) - 1):
        prev_session = sessions.get_group(skeys[i])
        next_session = sessions.get_group(skeys[i+1])
        last_prev = prev_session.iloc[-1]
        first_next = next_session.iloc[0]
        gap_hours = (first_next['timestamp'] - last_prev['timestamp']).total_seconds() / 3600
        session_edges.append({
            'user': uid[:60],
            'last_prev_score': last_prev['score'],
            'first_next_score': first_next['score'],
            'dscore_incubation': first_next['score'] - last_prev['score'],
            'gap_hours': gap_hours,
            'prev_session_len': len(prev_session),
            'prev_session': skeys[i],
            'next_session': skeys[i+1],
        })

edges_df = pd.DataFrame(session_edges)
print(f"\n[4d] Session transitions: {len(edges_df)}")
print(f"     Mean gap: {edges_df['gap_hours'].mean():.1f} hours")
print(f"     Mean Δscore (incubation): {edges_df['dscore_incubation'].mean():.3f}")
t_inc, p_inc = stats.ttest_1samp(edges_df['dscore_incubation'], 0)
print(f"     One-sample t-test vs 0: t={t_inc:.3f}, p={p_inc:.4f}")

# Compare to within-session consecutive pairs
within_pairs_dscore = within.values
print(f"     Within-session mean Δscore: {within_pairs_dscore.mean():.3f}")
print(f"     Between-session (incubation) mean Δscore: {edges_df['dscore_incubation'].mean():.3f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: distribution of incubation Δscore
axes[0].hist(edges_df['dscore_incubation'], bins=50, color='#FF6B6B', alpha=0.7, edgecolor='black')
axes[0].axvline(0, color='black', ls='--')
axes[0].axvline(edges_df['dscore_incubation'].mean(), color='blue', ls='-', lw=2,
                label=f'mean={edges_df["dscore_incubation"].mean():.2f}')
axes[0].set_xlabel('Δscore (first next session − last prev session)')
axes[0].set_ylabel('Count')
axes[0].set_title(f'Incubation Effect\nt={t_inc:.2f}, p={p_inc:.4f}')
axes[0].legend()

# Right: scatter gap_hours vs dscore_incubation
axes[1].scatter(edges_df['gap_hours'], edges_df['dscore_incubation'], alpha=0.3, s=15)
if len(edges_df) > 20:
    lw2 = lowess(edges_df['dscore_incubation'], edges_df['gap_hours'], frac=0.3)
    axes[1].plot(lw2[:, 0], lw2[:, 1], 'red', lw=3, label='LOWESS')
axes[1].axhline(0, color='gray', ls='--')
axes[1].set_xlabel('Break duration (hours)')
axes[1].set_ylabel('Δscore (incubation)')
axes[1].set_title('Score Change vs Break Duration')
axes[1].legend()

plt.tight_layout()
plt.savefig(f'{OUT}/B3d_incubation_effect.png', dpi=150)
plt.close()

# ── 5. Session-level analysis ──────────────────────────────────────
# Session means
session_stats = data.groupby(['user_key', 'session_id']).agg(
    mean_score=('score', 'mean'),
    max_score=('score', 'max'),
    n_attempts=('score', 'count'),
    first_score=('score', 'first'),
    last_score=('score', 'last'),
).reset_index()

# Cross-session improvement
session_stats['session_rank'] = session_stats.groupby('user_key')['session_id'].rank()

# Users with ≥3 sessions
multi_session_users = session_stats.groupby('user_key').filter(lambda x: len(x) >= 3)
print(f"\n[5] Users with ≥3 sessions: {multi_session_users['user_key'].nunique()}")

# Correlation: session_rank vs mean_score
if len(multi_session_users) > 10:
    r_sess, p_sess = stats.pearsonr(multi_session_users['session_rank'],
                                     multi_session_users['mean_score'])
    print(f"     Session rank vs mean score: r={r_sess:.4f}, p={p_sess:.4f}")

    # Mixed model approximation: per-user slope
    user_slopes = []
    for uid, ug in multi_session_users.groupby('user_key'):
        if len(ug) >= 3:
            slope, intercept, r, p, se = stats.linregress(ug['session_rank'], ug['mean_score'])
            user_slopes.append({'user': uid[:60], 'slope': slope, 'r': r, 'n_sessions': len(ug)})
    slopes_df = pd.DataFrame(user_slopes)
    print(f"     Per-user session slope: mean={slopes_df['slope'].mean():.3f}, "
          f"median={slopes_df['slope'].median():.3f}")
    t_slope, p_slope = stats.ttest_1samp(slopes_df['slope'], 0)
    print(f"     One-sample t-test on slopes: t={t_slope:.3f}, p={p_slope:.4f}")

# Within-session trend (fatigue)
within_session_slopes = []
for sk, sg in data.groupby('session_key'):
    if len(sg) >= 3:
        slope, _, r, p, _ = stats.linregress(sg['within_session_attempt'], sg['score'])
        within_session_slopes.append({
            'session': sk, 'slope': slope, 'r': r, 'n': len(sg)
        })
ws_slopes = pd.DataFrame(within_session_slopes)
print(f"\n     Sessions with ≥3 attempts: {len(ws_slopes)}")
print(f"     Within-session slope: mean={ws_slopes['slope'].mean():.3f}, "
      f"median={ws_slopes['slope'].median():.3f}")
t_ws, p_ws = stats.ttest_1samp(ws_slopes['slope'], 0)
print(f"     One-sample t-test on within-session slopes: t={t_ws:.3f}, p={p_ws:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Cross-session
if len(multi_session_users) > 10:
    axes[0].hist(slopes_df['slope'], bins=30, color='#4ECDC4', alpha=0.7, edgecolor='black')
    axes[0].axvline(0, color='black', ls='--')
    axes[0].axvline(slopes_df['slope'].mean(), color='red', lw=2,
                    label=f'mean={slopes_df["slope"].mean():.2f}')
    axes[0].set_xlabel('Slope (score per session)')
    axes[0].set_ylabel('Count (users)')
    axes[0].set_title(f'Cross-Session Learning\nt={t_slope:.2f}, p={p_slope:.4f}')
    axes[0].legend()

# Within-session fatigue
axes[1].hist(ws_slopes['slope'], bins=30, color='#FF6B6B', alpha=0.7, edgecolor='black')
axes[1].axvline(0, color='black', ls='--')
axes[1].axvline(ws_slopes['slope'].mean(), color='blue', lw=2,
                label=f'mean={ws_slopes["slope"].mean():.2f}')
axes[1].set_xlabel('Slope (score per attempt within session)')
axes[1].set_ylabel('Count (sessions)')
axes[1].set_title(f'Within-Session Trend (Fatigue?)\nt={t_ws:.2f}, p={p_ws:.4f}')
axes[1].legend()

plt.tight_layout()
plt.savefig(f'{OUT}/B3e_session_trends.png', dpi=150)
plt.close()

# ── 6. Time window comparison ──────────────────────────────────────
# Define windows
def time_window(m):
    if m < 10: return 'A: No break (<10 min)'
    elif 60 <= m < 360: return 'B: Short break (1-6h)'
    elif 480 <= m < 960: return 'C: Overnight (8-16h)'
    else: return None

pairs_df['time_window'] = pairs_df['interval_min'].apply(time_window)
tw_df = pairs_df[pairs_df['time_window'].notna()].copy()

tw_stats = tw_df.groupby('time_window')['dscore'].agg(['mean', 'std', 'count', 'median'])
print(f"\n[6] Time window comparison:")
print(tw_stats.to_string())

# ANOVA
groups = [g['dscore'].values for _, g in tw_df.groupby('time_window')]
f_stat, p_anova = stats.f_oneway(*groups)
print(f"     One-way ANOVA: F={f_stat:.3f}, p={p_anova:.4f}")

# Pairwise t-tests
windows = sorted(tw_df['time_window'].unique())
print("     Pairwise comparisons:")
for i in range(len(windows)):
    for j in range(i+1, len(windows)):
        g1 = tw_df[tw_df['time_window'] == windows[i]]['dscore']
        g2 = tw_df[tw_df['time_window'] == windows[j]]['dscore']
        t, p = stats.ttest_ind(g1, g2, equal_var=False)
        print(f"       {windows[i]} vs {windows[j]}: t={t:.3f}, p={p:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Bar plot
order = ['A: No break (<10 min)', 'B: Short break (1-6h)', 'C: Overnight (8-16h)']
colors = ['#4ECDC4', '#FFD93D', '#FF6B6B']
means = [tw_df[tw_df['time_window'] == w]['dscore'].mean() for w in order]
sems = [tw_df[tw_df['time_window'] == w]['dscore'].sem() for w in order]
ns = [len(tw_df[tw_df['time_window'] == w]) for w in order]

bars = axes[0].bar(range(3), means, yerr=[s*1.96 for s in sems], color=colors,
                   edgecolor='black', capsize=5)
axes[0].set_xticks(range(3))
axes[0].set_xticklabels([f'{w}\nn={n}' for w, n in zip(order, ns)], fontsize=9)
axes[0].axhline(0, color='gray', ls='--')
axes[0].set_ylabel('Mean Δscore (±95% CI)')
axes[0].set_title(f'B3f: Score Change by Break Type\nANOVA F={f_stat:.2f}, p={p_anova:.4f}')

# Violin plot
sns.violinplot(data=tw_df, x='time_window', y='dscore', order=order,
               palette=colors, ax=axes[1], inner='box', cut=0)
axes[1].axhline(0, color='gray', ls='--')
axes[1].set_xlabel('')
axes[1].set_ylabel('Δscore')
axes[1].set_title('Distribution of Score Changes by Break Type')
axes[1].tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig(f'{OUT}/B3f_time_windows.png', dpi=150)
plt.close()

# ── Summary ─────────────────────────────────────────────────────────
summary = f"""
=== B3: INCUBATION EFFECT — SUMMARY ===

DATA
  Total records: {len(df)}
  Specific-UA users with ≥5 attempts: {len(valid_users)}
  Consecutive pairs analyzed: {len(pairs_df)}

4b. WITHIN vs BETWEEN SESSION
  Within-session (<10 min): mean Δscore = {within.mean():.3f} (n={len(within)})
  Between-session (>6 hours): mean Δscore = {between.mean():.3f} (n={len(between)})
  Welch t = {t_stat:.3f}, p = {p_val:.4f}, Cohen's d = {cohens_d:.3f}

4c. REGRESSION (controlling for attempt number)
{model.summary2().tables[1].to_string()}
  Partial corr(log_interval, Δscore | attempt_n): r = {r_partial:.4f}, p = {p_partial:.4f}

4d. CLASSIC INCUBATION (first of new session vs last of previous)
  Session transitions: {len(edges_df)}
  Mean Δscore: {edges_df['dscore_incubation'].mean():.3f}
  t = {t_inc:.3f}, p = {p_inc:.4f}

5. SESSION-LEVEL TRENDS
  Cross-session slope: mean = {slopes_df['slope'].mean():.3f}, t = {t_slope:.3f}, p = {p_slope:.4f}
  Within-session slope: mean = {ws_slopes['slope'].mean():.3f}, t = {t_ws:.3f}, p = {p_ws:.4f}

6. TIME WINDOWS
{tw_stats.to_string()}
  ANOVA: F = {f_stat:.3f}, p = {p_anova:.4f}
"""
print(summary)
with open(f'{OUT}/B3_summary.txt', 'w') as f:
    f.write(summary)

print("All outputs saved to:", OUT)
