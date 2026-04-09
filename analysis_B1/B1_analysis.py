"""
B1: Learning Curve and Practice Effects Analysis for DAT-RU
"""
import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

OUT = '/Users/teo/Downloads/datcreativity/analysis_B1'

# ── Load data ──
df = pd.read_csv('/Users/teo/Downloads/DAT-RU Results - Sheet1 (1).csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(['user_agent', 'timestamp']).reset_index(drop=True)

# Add attempt number per user
df['attempt'] = df.groupby('user_agent').cumcount() + 1
user_counts = df.groupby('user_agent')['attempt'].max()

print(f"Total records: {len(df)}")
print(f"Unique user_agents: {len(user_counts)}")
print(f"Users with ≥2 attempts: {(user_counts >= 2).sum()}")
print(f"Users with ≥5 attempts: {(user_counts >= 5).sum()}")
print(f"Users with ≥10 attempts: {(user_counts >= 10).sum()}")
print(f"Users with ≥20 attempts: {(user_counts >= 20).sum()}")
print(f"Users with ≥50 attempts: {(user_counts >= 50).sum()}")
print(f"Max attempts by single user_agent: {user_counts.max()}")
print(f"Score range: {df['score'].min():.1f} – {df['score'].max():.1f}")
print(f"Score mean: {df['score'].mean():.2f}, SD: {df['score'].std():.2f}")
print()

# ══════════════════════════════════════════════════════════════
# STEP 2: Test-retest correlation (attempt 1 vs 2)
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 2: TEST-RETEST RELIABILITY")
print("=" * 60)

multi = df[df['user_agent'].isin(user_counts[user_counts >= 2].index)]
attempt1 = multi[multi['attempt'] == 1].set_index('user_agent')['score']
attempt2 = multi[multi['attempt'] == 2].set_index('user_agent')['score']
common = attempt1.index.intersection(attempt2.index)
s1, s2 = attempt1[common].values, attempt2[common].values

r, p = stats.pearsonr(s1, s2)
print(f"N users with ≥2 attempts: {len(common)}")
print(f"Pearson r(attempt1, attempt2) = {r:.4f}, p = {p:.2e}")

# Spearman
rho, p_rho = stats.spearmanr(s1, s2)
print(f"Spearman rho = {rho:.4f}, p = {p_rho:.2e}")

# ICC (two-way random, single measures, ICC(2,1))
n = len(s1)
grand_mean = np.mean(np.concatenate([s1, s2]))
ss_between = 2 * np.sum(((s1 + s2) / 2 - grand_mean) ** 2)
ss_within = np.sum((s1 - (s1 + s2) / 2) ** 2) + np.sum((s2 - (s1 + s2) / 2) ** 2)
ms_between = ss_between / (n - 1)
ms_within = ss_within / n
icc = (ms_between - ms_within) / (ms_between + ms_within)
print(f"ICC(2,1) ≈ {icc:.4f}")

# Mean difference
diff = s2 - s1
print(f"Mean score change (attempt2 - attempt1): {diff.mean():.3f} ± {diff.std():.3f}")
t_diff, p_diff = stats.ttest_rel(s2, s1)
print(f"Paired t-test: t = {t_diff:.3f}, p = {p_diff:.4f}")
print()

# Scatter plot
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(s1, s2, alpha=0.15, s=8, color='steelblue')
ax.plot([40, 100], [40, 100], 'k--', alpha=0.5, label='y=x')
ax.set_xlabel('Attempt 1 Score')
ax.set_ylabel('Attempt 2 Score')
ax.set_title(f'Test-Retest (N={len(common)})\nr = {r:.3f}, ICC = {icc:.3f}')
ax.legend()
fig.tight_layout()
fig.savefig(f'{OUT}/B1_test_retest_scatter.png', dpi=150)
plt.close()

# ══════════════════════════════════════════════════════════════
# STEP 3: Individual trajectories + group mean (≥5 attempts)
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 3: LEARNING CURVES")
print("=" * 60)

users5 = user_counts[user_counts >= 5].index
df5 = df[df['user_agent'].isin(users5)].copy()
print(f"Users with ≥5 attempts: {len(users5)}")
print(f"Records from these users: {len(df5)}")

# Cap attempt number for plotting
MAX_ATTEMPT_PLOT = 50

fig, ax = plt.subplots(figsize=(10, 6))

# Individual trajectories (thin gray)
for ua in users5:
    udf = df5[df5['user_agent'] == ua]
    udf_plot = udf[udf['attempt'] <= MAX_ATTEMPT_PLOT]
    ax.plot(udf_plot['attempt'], udf_plot['score'], color='gray', alpha=0.08, lw=0.5)

# Group mean ± SEM
mean_by_attempt = df5[df5['attempt'] <= MAX_ATTEMPT_PLOT].groupby('attempt')['score']
means = mean_by_attempt.mean()
sems = mean_by_attempt.sem()
counts = mean_by_attempt.count()

ax.plot(means.index, means.values, color='crimson', lw=2.5, label='Group mean')
ax.fill_between(means.index, means.values - sems.values, means.values + sems.values,
                color='crimson', alpha=0.2, label='±1 SEM')

ax2 = ax.twinx()
ax2.bar(counts.index, counts.values, alpha=0.1, color='steelblue', width=0.8)
ax2.set_ylabel('N users at this attempt', color='steelblue', alpha=0.5)
ax2.tick_params(axis='y', labelcolor='steelblue')

ax.set_xlabel('Attempt Number')
ax.set_ylabel('DAT Score')
ax.set_title(f'Learning Curves (N={len(users5)} users with ≥5 attempts)')
ax.legend(loc='upper left')
ax.set_xlim(0.5, MAX_ATTEMPT_PLOT + 0.5)
fig.tight_layout()
fig.savefig(f'{OUT}/B1_learning_curves.png', dpi=150)
plt.close()

# Also a zoomed-in version for first 20 attempts
fig, ax = plt.subplots(figsize=(9, 5.5))
for ua in users5:
    udf = df5[(df5['user_agent'] == ua) & (df5['attempt'] <= 20)]
    ax.plot(udf['attempt'], udf['score'], color='gray', alpha=0.1, lw=0.5)

m20 = df5[df5['attempt'] <= 20].groupby('attempt')['score']
means20 = m20.mean()
sems20 = m20.sem()
ax.plot(means20.index, means20.values, color='crimson', lw=2.5, label='Group mean')
ax.fill_between(means20.index, means20.values - sems20.values, means20.values + sems20.values,
                color='crimson', alpha=0.2)
# annotate N at each point
for i in [1, 5, 10, 15, 20]:
    if i in m20.count().index:
        ax.annotate(f'n={m20.count()[i]}', (i, means20[i]), fontsize=7, ha='center', va='bottom')
ax.set_xlabel('Attempt Number')
ax.set_ylabel('DAT Score')
ax.set_title('Learning Curves — First 20 Attempts')
ax.legend()
fig.tight_layout()
fig.savefig(f'{OUT}/B1_learning_curves_zoom20.png', dpi=150)
plt.close()

# Rolling mean for heavy users (≥30 attempts)
users30 = user_counts[user_counts >= 30].index
if len(users30) > 0:
    fig, ax = plt.subplots(figsize=(10, 6))
    for ua in users30:
        udf = df[df['user_agent'] == ua].copy()
        udf['rolling'] = udf['score'].rolling(10, min_periods=3).mean()
        ax.plot(udf['attempt'], udf['rolling'], alpha=0.4, lw=1)
    ax.set_xlabel('Attempt Number')
    ax.set_ylabel('Score (rolling mean, window=10)')
    ax.set_title(f'Heavy Users (≥30 attempts, N={len(users30)}): Rolling Mean Trajectories')
    fig.tight_layout()
    fig.savefig(f'{OUT}/B1_heavy_users_rolling.png', dpi=150)
    plt.close()
    print(f"Heavy users (≥30 attempts): {len(users30)}")

print()

# ══════════════════════════════════════════════════════════════
# STEP 4: Mixed-effects model / within-subject slopes
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 4: PRACTICE EFFECT — MIXED MODEL / WITHIN-SUBJECT SLOPES")
print("=" * 60)

# Try MixedLM first
try:
    import statsmodels.formula.api as smf
    from statsmodels.regression.mixed_linear_model import MixedLM

    # Use users with ≥5 attempts, cap at 100 attempts for stability
    df_model = df5[df5['attempt'] <= 100].copy()
    df_model['attempt_c'] = df_model['attempt'] - 1  # center at 0

    # Fit: score ~ attempt_c + (1 + attempt_c | user_agent)
    model = MixedLM.from_formula('score ~ attempt_c',
                                  groups='user_agent',
                                  re_formula='~attempt_c',
                                  data=df_model)
    result = model.fit(reml=True)
    print(result.summary())
    print()
    mixed_model_worked = True
except Exception as e:
    print(f"MixedLM failed: {e}")
    mixed_model_worked = False

# Fallback / complement: within-subject OLS slopes
print("\n--- Within-subject OLS slopes ---")
slopes = []
intercepts = []
for ua in users5:
    udf = df5[df5['user_agent'] == ua]
    if len(udf) >= 5:
        x = udf['attempt'].values
        y = udf['score'].values
        sl, inter, r_val, p_val, se = stats.linregress(x, y)
        slopes.append(sl)
        intercepts.append(inter)

slopes = np.array(slopes)
print(f"N users: {len(slopes)}")
print(f"Mean slope: {slopes.mean():.4f} ± {slopes.std():.4f}")
print(f"Median slope: {np.median(slopes):.4f}")
t_slope, p_slope = stats.ttest_1samp(slopes, 0)
print(f"One-sample t-test (slope ≠ 0): t = {t_slope:.3f}, p = {p_slope:.2e}")
print(f"% users with positive slope: {(slopes > 0).mean()*100:.1f}%")
print(f"% users with negative slope: {(slopes < 0).mean()*100:.1f}%")

# Wilcoxon as non-parametric complement
w_stat, w_p = stats.wilcoxon(slopes)
print(f"Wilcoxon signed-rank: W = {w_stat:.0f}, p = {w_p:.2e}")

# Distribution of slopes
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(slopes, bins=60, color='steelblue', edgecolor='white', alpha=0.8)
ax.axvline(0, color='red', ls='--', lw=1.5, label='zero')
ax.axvline(slopes.mean(), color='orange', ls='-', lw=2, label=f'mean = {slopes.mean():.3f}')
ax.axvline(np.median(slopes), color='green', ls=':', lw=2, label=f'median = {np.median(slopes):.3f}')
ax.set_xlabel('Within-subject slope (score per attempt)')
ax.set_ylabel('Count')
ax.set_title(f'Distribution of Within-Subject Slopes (N={len(slopes)})')
ax.legend()
fig.tight_layout()
fig.savefig(f'{OUT}/B1_slope_distribution.png', dpi=150)
plt.close()

print()

# ══════════════════════════════════════════════════════════════
# STEP 5: Personal best analysis
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 5: PERSONAL BEST ANALYSIS")
print("=" * 60)

for min_att in [5, 10, 20]:
    users_n = user_counts[user_counts >= min_att].index
    if len(users_n) < 5:
        continue

    # For each user, cumulative max as function of attempt
    max_attempt_n = min(min_att * 3, 100)
    pb_matrix = []
    for ua in users_n:
        udf = df[df['user_agent'] == ua].head(max_attempt_n)
        cummax = udf['score'].cummax().values
        # Normalize by user's overall personal best
        user_pb = udf['score'].max()
        if user_pb > 0:
            pb_norm = cummax / user_pb
            pb_matrix.append(pb_norm)

    # Pad to same length and average
    max_len = max(len(x) for x in pb_matrix)
    pb_padded = np.full((len(pb_matrix), max_len), np.nan)
    for i, pb in enumerate(pb_matrix):
        pb_padded[i, :len(pb)] = pb

    pb_mean = np.nanmean(pb_padded, axis=0)
    pb_count = np.sum(~np.isnan(pb_padded), axis=0)

    # Find 95% of ceiling
    attempts_to_95 = np.where(pb_mean >= 0.95)[0]
    if len(attempts_to_95) > 0:
        att_95 = attempts_to_95[0] + 1
        print(f"Users ≥{min_att} attempts (N={len(users_n)}): reach 95% of personal best at attempt {att_95}")
    else:
        print(f"Users ≥{min_att} attempts (N={len(users_n)}): never reach 95% within {max_attempt_n} attempts")

# Plot for ≥10 attempts users
users10 = user_counts[user_counts >= 10].index
if len(users10) >= 5:
    max_att_plot = 40
    pb_matrix10 = []
    for ua in users10:
        udf = df[df['user_agent'] == ua].head(max_att_plot)
        cummax = udf['score'].cummax().values
        user_pb = df[df['user_agent'] == ua]['score'].max()  # overall PB
        if user_pb > 0:
            pb_matrix10.append(cummax / user_pb)

    max_len10 = max(len(x) for x in pb_matrix10)
    pb_pad10 = np.full((len(pb_matrix10), max_len10), np.nan)
    for i, pb in enumerate(pb_matrix10):
        pb_pad10[i, :len(pb)] = pb

    pb_mean10 = np.nanmean(pb_pad10, axis=0)
    pb_sem10 = np.nanstd(pb_pad10, axis=0) / np.sqrt(np.sum(~np.isnan(pb_pad10), axis=0))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(1, len(pb_mean10) + 1)
    ax.plot(x, pb_mean10, color='crimson', lw=2.5)
    ax.fill_between(x, pb_mean10 - pb_sem10, pb_mean10 + pb_sem10, color='crimson', alpha=0.2)
    ax.axhline(0.95, color='gray', ls='--', alpha=0.5, label='95% of personal best')
    ax.axhline(1.0, color='gray', ls=':', alpha=0.3)
    att95_idx = np.where(pb_mean10 >= 0.95)[0]
    if len(att95_idx) > 0:
        ax.axvline(att95_idx[0] + 1, color='steelblue', ls='--', alpha=0.5,
                   label=f'95% reached at attempt {att95_idx[0]+1}')
    ax.set_xlabel('Attempt Number')
    ax.set_ylabel('Fraction of Personal Best')
    ax.set_title(f'Personal Best Accumulation (N={len(pb_matrix10)} users with ≥10 attempts)')
    ax.set_ylim(0.7, 1.02)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f'{OUT}/B1_personal_best.png', dpi=150)
    plt.close()

print()

# ══════════════════════════════════════════════════════════════
# STEP 6: Changepoint detection for heavy users
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 6: CHANGEPOINT DETECTION (≥50 ATTEMPTS)")
print("=" * 60)

users50 = user_counts[user_counts >= 50].index
print(f"Users with ≥50 attempts: {len(users50)}")

def fit_piecewise(x, y):
    """Fit piecewise linear with 1 changepoint. Returns (cp, slope1, slope2, residual)."""
    best_rss = np.inf
    best_cp = None
    best_params = None

    for cp in range(5, len(x) - 5):
        x1, y1 = x[:cp], y[:cp]
        x2, y2 = x[cp:], y[cp:]

        if len(x1) < 3 or len(x2) < 3:
            continue

        sl1, int1, _, _, _ = stats.linregress(x1, y1)
        sl2, int2, _, _, _ = stats.linregress(x2, y2)

        pred1 = int1 + sl1 * x1
        pred2 = int2 + sl2 * x2
        rss = np.sum((y1 - pred1)**2) + np.sum((y2 - pred2)**2)

        if rss < best_rss:
            best_rss = rss
            best_cp = x[cp]
            best_params = (sl1, sl2, int1, int2)

    return best_cp, best_params, best_rss

if len(users50) >= 3:
    changepoints = []
    slope_changes = []
    for ua in users50:
        udf = df[df['user_agent'] == ua].copy()
        x = udf['attempt'].values.astype(float)
        y = udf['score'].values.astype(float)

        # smooth first
        from scipy.ndimage import uniform_filter1d
        y_smooth = uniform_filter1d(y, size=min(10, len(y)//3))

        cp, params, rss = fit_piecewise(x, y_smooth)
        if cp is not None and params is not None:
            changepoints.append(cp)
            slope_changes.append((params[0], params[1]))  # slope1, slope2

    changepoints = np.array(changepoints)
    print(f"Changepoints detected for {len(changepoints)} users")
    print(f"Mean changepoint: attempt {changepoints.mean():.1f} ± {changepoints.std():.1f}")
    print(f"Median changepoint: attempt {np.median(changepoints):.1f}")

    # Classify: improvement→plateau/decline
    n_improve_then_decline = sum(1 for s1, s2 in slope_changes if s1 > 0 and s2 < s1)
    n_decline_then_improve = sum(1 for s1, s2 in slope_changes if s1 < s2 and s2 > 0)
    print(f"Improvement → plateau/decline: {n_improve_then_decline}/{len(slope_changes)}")
    print(f"Decline → improvement: {n_decline_then_improve}/{len(slope_changes)}")

    # Plot changepoint distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(changepoints, bins=20, color='steelblue', edgecolor='white')
    axes[0].axvline(changepoints.mean(), color='red', ls='--', label=f'mean={changepoints.mean():.0f}')
    axes[0].set_xlabel('Changepoint (attempt number)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Distribution of Changepoints')
    axes[0].legend()

    # Slope before vs after changepoint
    s1s = [s[0] for s in slope_changes]
    s2s = [s[1] for s in slope_changes]
    axes[1].scatter(s1s, s2s, alpha=0.5, s=30, color='steelblue')
    lim = max(abs(min(s1s + s2s)), abs(max(s1s + s2s))) * 1.1
    axes[1].set_xlim(-lim, lim)
    axes[1].set_ylim(-lim, lim)
    axes[1].plot([-lim, lim], [-lim, lim], 'k--', alpha=0.3)
    axes[1].axhline(0, color='gray', alpha=0.3)
    axes[1].axvline(0, color='gray', alpha=0.3)
    axes[1].set_xlabel('Slope before changepoint')
    axes[1].set_ylabel('Slope after changepoint')
    axes[1].set_title('Slope Change at Changepoint')
    fig.tight_layout()
    fig.savefig(f'{OUT}/B1_changepoints.png', dpi=150)
    plt.close()
else:
    print("Not enough heavy users for changepoint analysis")

print()

# ══════════════════════════════════════════════════════════════
# STEP 7: Strategy analysis via Jaccard similarity
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 7: JACCARD STRATEGY ANALYSIS")
print("=" * 60)

def parse_words(w):
    """Parse comma-separated Russian words."""
    if pd.isna(w):
        return set()
    return set(word.strip().lower() for word in str(w).split(',') if word.strip())

def jaccard(s1, s2):
    if len(s1) == 0 and len(s2) == 0:
        return 1.0
    inter = len(s1 & s2)
    union = len(s1 | s2)
    return inter / union if union > 0 else 0.0

# Compute Jaccard for consecutive attempts
jacc_records = []
for ua in users5:
    udf = df5[df5['user_agent'] == ua].sort_values('attempt')
    words_list = [parse_words(w) for w in udf['words'].values]
    scores = udf['score'].values
    attempts = udf['attempt'].values

    for i in range(1, len(words_list)):
        j = jaccard(words_list[i-1], words_list[i])
        score_change = scores[i] - scores[i-1]
        jacc_records.append({
            'user_agent': ua,
            'attempt': attempts[i],
            'jaccard': j,
            'score_change': score_change,
            'score': scores[i],
            'prev_score': scores[i-1]
        })

jacc_df = pd.DataFrame(jacc_records)
print(f"Consecutive attempt pairs: {len(jacc_df)}")
print(f"Mean Jaccard: {jacc_df['jaccard'].mean():.4f}")
print(f"Median Jaccard: {jacc_df['jaccard'].median():.4f}")
print(f"Jaccard = 0 (complete restart): {(jacc_df['jaccard'] == 0).mean()*100:.1f}%")
print(f"Jaccard > 0.5 (optimizer, >3 shared words): {(jacc_df['jaccard'] > 0.5).mean()*100:.1f}%")

# Correlation: Jaccard with score change
r_j, p_j = stats.pearsonr(jacc_df['jaccard'], jacc_df['score_change'])
print(f"\nJaccard vs score_change: r = {r_j:.4f}, p = {p_j:.2e}")

# Mean user-level Jaccard
user_jacc = jacc_df.groupby('user_agent').agg(
    mean_jaccard=('jaccard', 'mean'),
    mean_score_change=('score_change', 'mean'),
    n_pairs=('jaccard', 'count')
).reset_index()

# User-level: classify strategies
user_jacc['strategy'] = pd.cut(user_jacc['mean_jaccard'],
                                bins=[-0.01, 0.1, 0.3, 1.01],
                                labels=['Restarter', 'Mixed', 'Optimizer'])

print(f"\nUser-level strategy distribution:")
print(user_jacc['strategy'].value_counts().to_string())

# Mean score improvement by strategy
for strat in ['Restarter', 'Mixed', 'Optimizer']:
    sub = user_jacc[user_jacc['strategy'] == strat]
    if len(sub) > 0:
        print(f"  {strat}: mean score change = {sub['mean_score_change'].mean():.3f} (N={len(sub)})")

# Compute user-level slopes and correlate with Jaccard
user_slopes = {}
for ua in users5:
    udf = df5[df5['user_agent'] == ua]
    if len(udf) >= 5:
        sl, _, _, _, _ = stats.linregress(udf['attempt'].values, udf['score'].values)
        user_slopes[ua] = sl

user_jacc['slope'] = user_jacc['user_agent'].map(user_slopes)
valid = user_jacc.dropna(subset=['slope'])
r_js, p_js = stats.pearsonr(valid['mean_jaccard'], valid['slope'])
print(f"\nUser-level: mean_jaccard vs slope: r = {r_js:.4f}, p = {p_js:.2e}")

# Plots
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 1. Jaccard distribution
axes[0].hist(jacc_df['jaccard'], bins=30, color='steelblue', edgecolor='white')
axes[0].set_xlabel('Jaccard Similarity (consecutive attempts)')
axes[0].set_ylabel('Count')
axes[0].set_title('Word Set Overlap Between Attempts')
axes[0].axvline(jacc_df['jaccard'].mean(), color='red', ls='--', label=f'mean={jacc_df["jaccard"].mean():.2f}')
axes[0].legend()

# 2. Jaccard vs score change (binned)
bins_j = pd.cut(jacc_df['jaccard'], bins=10)
binned = jacc_df.groupby(bins_j)['score_change'].agg(['mean', 'sem', 'count'])
x_centers = [(b.left + b.right) / 2 for b in binned.index]
axes[1].bar(x_centers, binned['mean'], width=0.08, color='steelblue', alpha=0.7)
axes[1].errorbar(x_centers, binned['mean'], yerr=binned['sem'], fmt='none', color='black', capsize=3)
axes[1].axhline(0, color='gray', ls='--', alpha=0.5)
axes[1].set_xlabel('Jaccard Similarity')
axes[1].set_ylabel('Mean Score Change')
axes[1].set_title(f'Score Change by Word Overlap\nr={r_j:.3f}')

# 3. Strategy and slope
for strat, color in zip(['Restarter', 'Mixed', 'Optimizer'], ['#e74c3c', '#f39c12', '#2ecc71']):
    sub = valid[valid['strategy'] == strat]
    axes[2].scatter(sub['mean_jaccard'], sub['slope'], alpha=0.4, s=20, color=color, label=strat)
axes[2].axhline(0, color='gray', ls='--', alpha=0.5)
axes[2].set_xlabel('Mean Jaccard (user-level)')
axes[2].set_ylabel('Within-subject Slope')
axes[2].set_title(f'Strategy vs Learning Rate\nr={r_js:.3f}')
axes[2].legend()

fig.tight_layout()
fig.savefig(f'{OUT}/B1_jaccard_strategy.png', dpi=150)
plt.close()

# ══════════════════════════════════════════════════════════════
# Additional: Jaccard over time — does strategy change with practice?
# ══════════════════════════════════════════════════════════════
jacc_by_attempt = jacc_df[jacc_df['attempt'] <= 30].groupby('attempt')['jaccard'].agg(['mean', 'sem', 'count'])
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(jacc_by_attempt.index, jacc_by_attempt['mean'], color='crimson', lw=2)
ax.fill_between(jacc_by_attempt.index,
                jacc_by_attempt['mean'] - jacc_by_attempt['sem'],
                jacc_by_attempt['mean'] + jacc_by_attempt['sem'],
                color='crimson', alpha=0.2)
ax.set_xlabel('Attempt Number')
ax.set_ylabel('Mean Jaccard with Previous Attempt')
ax.set_title('Word Overlap Strategy Over Practice')
fig.tight_layout()
fig.savefig(f'{OUT}/B1_jaccard_over_time.png', dpi=150)
plt.close()

print("\n" + "=" * 60)
print("ALL DONE. Plots saved to:", OUT)
print("=" * 60)
