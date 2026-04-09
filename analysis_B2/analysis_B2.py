"""
B2: Practice effects — random sampling vs genuine learning
"""
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import re
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load & classify UAs ──────────────────────────────────────────
df = pd.read_csv("/Users/teo/Downloads/DAT-RU Results - Sheet1 (1).csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

def extract_device(ua):
    """Extract specific device model from UA string."""
    if not isinstance(ua, str):
        return None
    # iPhone models: FBDV/iPhone17,1 or just iPhone\d+,\d+
    m = re.search(r'iPhone(\d+,\d+)', ua)
    if m:
        return f"iPhone{m.group(1)}"
    # Samsung: SM-XXXX
    m = re.search(r'(SM-[A-Z]\w+)', ua)
    if m:
        return m.group(1)
    # Pixel
    m = re.search(r'(Pixel\s*\w+)', ua)
    if m:
        return m.group(1)
    # Xiaomi / Redmi / POCO with actual model numbers
    m = re.search(r'((?:Redmi|POCO|Mi)\s*\w+)', ua)
    if m:
        return m.group(1)
    # Other Android with actual model (not just "K")
    m = re.search(r'Android\s+\d+;\s+([A-Z][A-Za-z0-9]+[-\s][A-Za-z0-9]+)', ua)
    if m and m.group(1).strip() != 'K':
        return m.group(1).strip()
    # Huawei
    m = re.search(r'((?:VOG|ELE|MAR|CLT|LYA|ANA|NOH|OCE)\-\w+)', ua)
    if m:
        return m.group(1)
    return None

df['device'] = df['user_agent'].apply(extract_device)
df['ua_type'] = df['device'].apply(lambda x: 'specific' if x is not None else 'generic')

print(f"Total records: {len(df)}")
print(f"Specific UA: {(df['ua_type']=='specific').sum()} ({(df['ua_type']=='specific').mean()*100:.1f}%)")
print(f"Generic UA: {(df['ua_type']=='generic').sum()} ({(df['ua_type']=='generic').mean()*100:.1f}%)")

# ── Focus on specific-UA users ──────────────────────────────────────
spec = df[df['ua_type'] == 'specific'].copy()
# Use device as user ID proxy
user_counts = spec.groupby('device').size()
print(f"\nSpecific-UA unique devices: {len(user_counts)}")
print(f"Devices with >=10 attempts: {(user_counts >= 10).sum()}")
print(f"Devices with >=5 attempts: {(user_counts >= 5).sum()}")

# Use >=5 for more power, but analyze >=10 subset too
MIN_ATTEMPTS = 5
qualifying_devices = user_counts[user_counts >= MIN_ATTEMPTS].index.tolist()
print(f"\nUsing MIN_ATTEMPTS={MIN_ATTEMPTS}: {len(qualifying_devices)} users")

# Add attempt number per user
spec = spec.sort_values(['device', 'timestamp'])
spec['attempt'] = spec.groupby('device').cumcount() + 1

qual = spec[spec['device'].isin(qualifying_devices)].copy()
print(f"Qualifying records: {len(qual)}")

# ── 2. Per-user analysis ────────────────────────────────────────────
np.random.seed(42)
N_SIM = 1000
MAX_ATTEMPT = 30  # cap for visualization

results = []

for dev in qualifying_devices:
    user_data = qual[qual['device'] == dev].sort_values('attempt')
    scores = user_data['score'].values
    N = len(scores)
    mu = np.mean(scores)
    sigma = np.std(scores, ddof=1)

    # Real personal best curve
    real_pb = np.maximum.accumulate(scores)

    # Simulated i.i.d. personal best
    if sigma > 0:
        sim_draws = np.random.normal(mu, sigma, size=(N_SIM, N))
    else:
        sim_draws = np.full((N_SIM, N), mu)
    sim_pb = np.maximum.accumulate(sim_draws, axis=1)
    sim_pb_mean = sim_pb.mean(axis=0)
    sim_pb_lo = np.percentile(sim_pb, 2.5, axis=0)
    sim_pb_hi = np.percentile(sim_pb, 97.5, axis=0)

    # Score slope (OLS)
    attempts_arr = np.arange(1, N+1)
    slope_real, intercept, r_val, p_val, se = stats.linregress(attempts_arr, scores)

    # Simulated slopes
    sim_slopes = []
    for i in range(N_SIM):
        s = sim_draws[i]
        sl, _, _, _, _ = stats.linregress(attempts_arr, s)
        sim_slopes.append(sl)
    sim_slopes = np.array(sim_slopes)

    # Is real slope above 95th percentile of i.i.d. slopes?
    slope_percentile = np.mean(sim_slopes < slope_real)

    # Warm-up test: first 3 vs attempts 4-6
    if N >= 6:
        early = scores[:3].mean()
        later = scores[3:6].mean()
        warmup_diff = later - early
    else:
        early = scores[:N//2].mean()
        later = scores[N//2:].mean()
        warmup_diff = later - early

    results.append({
        'device': dev,
        'N': N,
        'mu': mu,
        'sigma': sigma,
        'slope_real': slope_real,
        'slope_p': p_val,
        'sim_slope_mean': sim_slopes.mean(),
        'sim_slope_std': sim_slopes.std(),
        'slope_percentile': slope_percentile,
        'real_pb_final': real_pb[-1],
        'sim_pb_final_mean': sim_pb_mean[-1],
        'pb_diff': real_pb[-1] - sim_pb_mean[-1],
        'warmup_diff': warmup_diff,
        'real_pb': real_pb,
        'sim_pb_mean': sim_pb_mean,
        'sim_pb_lo': sim_pb_lo,
        'sim_pb_hi': sim_pb_hi,
        'scores': scores,
    })

res_df = pd.DataFrame(results)
print(f"\n{'='*60}")
print(f"PER-USER SUMMARY (N={len(res_df)} users, min {MIN_ATTEMPTS} attempts)")
print(f"{'='*60}")
print(f"Mean real slope:      {res_df['slope_real'].mean():.4f} +/- {res_df['slope_real'].std():.4f}")
print(f"Mean simulated slope: {res_df['sim_slope_mean'].mean():.4f} +/- {res_df['sim_slope_mean'].std():.4f}")
print(f"Mean PB diff (real-sim): {res_df['pb_diff'].mean():.2f}")
print(f"Mean warmup diff:     {res_df['warmup_diff'].mean():.2f}")

# ── 3. Group-level personal best curves ─────────────────────────────
# Align all users to common attempt axis (pad with NaN for shorter series)
max_n = min(MAX_ATTEMPT, qual.groupby('device').size().max())

# Only use users with enough attempts for each checkpoint
checkpoints = [5, 10, 15, 20]

print(f"\n{'='*60}")
print(f"GROUP-LEVEL PERSONAL BEST: REAL vs I.I.D.")
print(f"{'='*60}")

# Build aligned matrices
real_pb_matrix = []
sim_pb_matrix = []

for r in results:
    n = min(len(r['real_pb']), max_n)
    row_real = np.full(max_n, np.nan)
    row_sim = np.full(max_n, np.nan)
    row_real[:n] = r['real_pb'][:n]
    row_sim[:n] = r['sim_pb_mean'][:n]
    real_pb_matrix.append(row_real)
    sim_pb_matrix.append(row_sim)

real_pb_matrix = np.array(real_pb_matrix)
sim_pb_matrix = np.array(sim_pb_matrix)

# Statistical tests at checkpoints
print(f"\nPaired t-tests (real PB - simulated PB):")
for cp in checkpoints:
    if cp > max_n:
        continue
    idx = cp - 1
    mask = ~np.isnan(real_pb_matrix[:, idx]) & ~np.isnan(sim_pb_matrix[:, idx])
    n_users = mask.sum()
    if n_users < 3:
        continue
    real_vals = real_pb_matrix[mask, idx]
    sim_vals = sim_pb_matrix[mask, idx]
    diff = real_vals - sim_vals
    t_stat, p_val = stats.ttest_rel(real_vals, sim_vals)
    # Also Wilcoxon
    if n_users >= 10:
        w_stat, w_p = stats.wilcoxon(diff)
    else:
        w_stat, w_p = np.nan, np.nan
    print(f"  Attempt {cp:2d}: N={n_users:3d}, diff={diff.mean():.2f} +/- {diff.std():.2f}, "
          f"t={t_stat:.3f}, p={p_val:.4f}, Wilcoxon p={w_p:.4f}")

# ── 4. Score trajectory / slope analysis ────────────────────────────
print(f"\n{'='*60}")
print(f"SLOPE ANALYSIS")
print(f"{'='*60}")

# Real slope distribution
real_slopes = res_df['slope_real'].values
print(f"Real slopes: mean={real_slopes.mean():.4f}, median={np.median(real_slopes):.4f}")
print(f"  % positive: {(real_slopes > 0).mean()*100:.1f}%")

# One-sample t-test: is mean slope > 0?
t_slope, p_slope = stats.ttest_1samp(real_slopes, 0)
print(f"  One-sample t-test (slope > 0): t={t_slope:.3f}, p={p_slope:.4f} (two-sided), p/2={p_slope/2:.4f} (one-sided)")

# Compare real slopes to what i.i.d. predicts
print(f"\nFraction of users with slope above 95th pct of their i.i.d. null: "
      f"{(res_df['slope_percentile'] > 0.95).mean()*100:.1f}%")
print(f"Fraction of users with slope above 90th pct of their i.i.d. null: "
      f"{(res_df['slope_percentile'] > 0.90).mean()*100:.1f}%")

# Expected under null: 5% and 10%
n_above_95 = (res_df['slope_percentile'] > 0.95).sum()
n_total = len(res_df)
# Binomial test: is the fraction above 5% significantly?
binom_p = 1 - stats.binom.cdf(n_above_95 - 1, n_total, 0.05)
print(f"  Binomial test (>5% expected): {n_above_95}/{n_total} above 95th pct, p={binom_p:.4f}")

# ── 5. Warm-up test ─────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"WARM-UP TEST (first 3 vs attempts 4-6)")
print(f"{'='*60}")

warmup_diffs = res_df['warmup_diff'].values
t_wu, p_wu = stats.ttest_1samp(warmup_diffs, 0)
print(f"Mean diff (later - early): {warmup_diffs.mean():.2f} +/- {warmup_diffs.std():.2f}")
print(f"t={t_wu:.3f}, p={p_wu:.4f} (two-sided), p/2={p_wu/2:.4f} (one-sided)")
print(f"% users with positive warmup: {(warmup_diffs > 0).mean()*100:.1f}%")

# Wilcoxon signed-rank
if len(warmup_diffs) >= 10:
    w_wu, wp_wu = stats.wilcoxon(warmup_diffs)
    print(f"Wilcoxon: W={w_wu:.1f}, p={wp_wu:.4f}")

# ── 6. Interpretation ──────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"INTERPRETATION")
print(f"{'='*60}")

pb_diffs = res_df['pb_diff'].values
mean_pb_diff = pb_diffs.mean()
if mean_pb_diff > 1:
    interp = "GENUINE LEARNING: real personal bests exceed i.i.d. prediction"
elif mean_pb_diff < -1:
    interp = "FATIGUE/DEPLETION: real personal bests fall below i.i.d. prediction"
else:
    interp = "RANDOM SAMPLING: real personal bests match i.i.d. prediction"
print(f"PB difference: {mean_pb_diff:.2f} -> {interp}")

slope_sig = p_slope / 2 < 0.05 and real_slopes.mean() > 0
print(f"Slope evidence: mean={real_slopes.mean():.4f}, p(one-sided)={p_slope/2:.4f} -> {'significant' if slope_sig else 'not significant'}")

warmup_sig = p_wu / 2 < 0.05 and warmup_diffs.mean() > 0
print(f"Warmup evidence: diff={warmup_diffs.mean():.2f}, p(one-sided)={p_wu/2:.4f} -> {'significant' if warmup_sig else 'not significant'}")


# ══════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════
OUT = "/Users/teo/Downloads/datcreativity/analysis_B2"

# ── Plot 1: Group-level personal best curves ────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

attempts_axis = np.arange(1, max_n + 1)
# Count how many users contribute at each attempt
n_per_attempt = np.sum(~np.isnan(real_pb_matrix), axis=0)

real_mean = np.nanmean(real_pb_matrix, axis=0)
real_se = np.nanstd(real_pb_matrix, axis=0) / np.sqrt(n_per_attempt)
sim_mean = np.nanmean(sim_pb_matrix, axis=0)
sim_se = np.nanstd(sim_pb_matrix, axis=0) / np.sqrt(n_per_attempt)

# Only plot where we have at least 5 users
mask_plot = n_per_attempt >= 5

ax.plot(attempts_axis[mask_plot], real_mean[mask_plot], 'b-o', lw=2, ms=4, label='Real personal best')
ax.fill_between(attempts_axis[mask_plot],
                real_mean[mask_plot] - 1.96*real_se[mask_plot],
                real_mean[mask_plot] + 1.96*real_se[mask_plot], alpha=0.2, color='blue')

ax.plot(attempts_axis[mask_plot], sim_mean[mask_plot], 'r--s', lw=2, ms=4, label='Simulated i.i.d. personal best')
ax.fill_between(attempts_axis[mask_plot],
                sim_mean[mask_plot] - 1.96*sim_se[mask_plot],
                sim_mean[mask_plot] + 1.96*sim_se[mask_plot], alpha=0.2, color='red')

# Add N annotation
for cp in [1, 5, 10, 15, 20, 25, 30]:
    if cp <= max_n and cp-1 < len(n_per_attempt) and n_per_attempt[cp-1] >= 5:
        ax.annotate(f'n={int(n_per_attempt[cp-1])}', (cp, real_mean[cp-1]),
                    textcoords="offset points", xytext=(0, 12), fontsize=7, color='gray')

ax.set_xlabel('Attempt number', fontsize=12)
ax.set_ylabel('Personal best DAT score', fontsize=12)
ax.set_title('Real vs I.I.D. Simulated Personal Best Curves\n(Specific-UA users, group average)', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/B2_personal_best_curves.png", dpi=150)
plt.close()
print(f"\nSaved: B2_personal_best_curves.png")

# ── Plot 2: Slope distributions ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Real slopes histogram
ax = axes[0]
ax.hist(real_slopes, bins=30, alpha=0.7, color='steelblue', edgecolor='black', density=True)
ax.axvline(0, color='red', ls='--', lw=2, label='Null (slope=0)')
ax.axvline(real_slopes.mean(), color='blue', ls='-', lw=2, label=f'Mean={real_slopes.mean():.3f}')
ax.set_xlabel('Score slope (per attempt)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Distribution of real score slopes', fontsize=12)
ax.legend()

# Real vs simulated slope comparison
ax = axes[1]
# Pool all simulated slopes from all users
all_sim_slopes = []
for r in results:
    # Regenerate sim slopes for this user
    N = r['N']
    sim_draws = np.random.normal(r['mu'], max(r['sigma'], 0.01), size=(200, N))
    for i in range(200):
        sl, _, _, _, _ = stats.linregress(np.arange(1, N+1), sim_draws[i])
        all_sim_slopes.append(sl)
all_sim_slopes = np.array(all_sim_slopes)

ax.hist(all_sim_slopes, bins=50, alpha=0.5, color='red', density=True, label='Simulated i.i.d. slopes')
ax.hist(real_slopes, bins=30, alpha=0.5, color='blue', density=True, label='Real slopes')
ax.axvline(0, color='black', ls='--', lw=1)
ax.set_xlabel('Score slope (per attempt)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Real vs simulated slope distributions', fontsize=12)
ax.legend()

plt.tight_layout()
plt.savefig(f"{OUT}/B2_slope_distributions.png", dpi=150)
plt.close()
print(f"Saved: B2_slope_distributions.png")

# ── Plot 3: Warm-up effect ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(warmup_diffs, bins=30, alpha=0.7, color='teal', edgecolor='black')
ax.axvline(0, color='red', ls='--', lw=2, label='No warm-up (diff=0)')
ax.axvline(warmup_diffs.mean(), color='blue', ls='-', lw=2,
           label=f'Mean diff={warmup_diffs.mean():.2f}')
ax.set_xlabel('Score difference (attempts 4-6 minus 1-3)', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title(f'Warm-up test: later minus early attempts\n'
             f'p={p_wu/2:.4f} (one-sided), N={len(warmup_diffs)} users', fontsize=12)
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/B2_warmup_effect.png", dpi=150)
plt.close()
print(f"Saved: B2_warmup_effect.png")

# ── Plot 4: Individual examples ─────────────────────────────────────
# Show 6 users with most attempts
top_users = sorted(results, key=lambda x: x['N'], reverse=True)[:6]

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for idx, (ax, r) in enumerate(zip(axes.flat, top_users)):
    n = r['N']
    x = np.arange(1, n+1)

    # Raw scores
    ax.plot(x, r['scores'], 'gray', alpha=0.5, lw=1, label='Raw scores')
    ax.scatter(x, r['scores'], c='gray', s=15, alpha=0.5)

    # Real PB
    ax.plot(x, r['real_pb'], 'b-', lw=2, label='Real PB')

    # Simulated PB
    ax.plot(x, r['sim_pb_mean'], 'r--', lw=2, label='Sim PB (i.i.d.)')
    ax.fill_between(x, r['sim_pb_lo'], r['sim_pb_hi'], alpha=0.15, color='red')

    ax.set_title(f"{r['device'][:20]} (N={n}, slope={r['slope_real']:.3f})", fontsize=9)
    ax.set_xlabel('Attempt')
    ax.set_ylabel('DAT score')
    if idx == 0:
        ax.legend(fontsize=7)

plt.suptitle('Individual users: real vs i.i.d. personal best curves', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT}/B2_individual_examples.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: B2_individual_examples.png")

# ── Plot 5: Summary dashboard ──────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# PB diff distribution
ax = axes[0, 0]
ax.hist(pb_diffs, bins=25, alpha=0.7, color='purple', edgecolor='black')
ax.axvline(0, color='red', ls='--', lw=2)
ax.axvline(pb_diffs.mean(), color='blue', ls='-', lw=2, label=f'Mean={pb_diffs.mean():.2f}')
ax.set_xlabel('PB difference (real - simulated)')
ax.set_ylabel('Count')
ax.set_title('Personal best: real minus i.i.d.')
ax.legend()

# Slope percentile distribution
ax = axes[0, 1]
ax.hist(res_df['slope_percentile'], bins=20, alpha=0.7, color='orange', edgecolor='black')
ax.axvline(0.5, color='red', ls='--', lw=2, label='Expected median')
ax.axvline(0.95, color='green', ls=':', lw=2, label='95th percentile')
ax.set_xlabel('Percentile of real slope in i.i.d. null')
ax.set_ylabel('Count')
ax.set_title(f'Slope percentile distribution\n(>{0.95:.0%}: {(res_df["slope_percentile"]>0.95).mean()*100:.1f}% of users)')
ax.legend(fontsize=9)

# N attempts distribution
ax = axes[1, 0]
ax.hist(res_df['N'], bins=range(MIN_ATTEMPTS, res_df['N'].max()+2), alpha=0.7, color='green', edgecolor='black')
ax.set_xlabel('Number of attempts')
ax.set_ylabel('Count')
ax.set_title(f'Attempts per user (N={len(res_df)} users)')

# Slope vs N attempts
ax = axes[1, 1]
ax.scatter(res_df['N'], res_df['slope_real'], alpha=0.5, s=30, c='steelblue')
ax.axhline(0, color='red', ls='--', lw=1)
r_corr, p_corr = stats.spearmanr(res_df['N'], res_df['slope_real'])
ax.set_xlabel('Number of attempts')
ax.set_ylabel('Score slope')
ax.set_title(f'Slope vs experience\n(Spearman r={r_corr:.3f}, p={p_corr:.4f})')

plt.suptitle('B2 Summary: Practice Effects Analysis', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT}/B2_summary_dashboard.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: B2_summary_dashboard.png")

# ── Also do the analysis with >=10 attempts ─────────────────────────
print(f"\n{'='*60}")
print(f"ROBUSTNESS: USERS WITH >=10 ATTEMPTS")
print(f"{'='*60}")
res10 = res_df[res_df['N'] >= 10]
if len(res10) > 0:
    print(f"N users: {len(res10)}")
    slopes10 = res10['slope_real'].values
    t10, p10 = stats.ttest_1samp(slopes10, 0)
    print(f"Mean slope: {slopes10.mean():.4f}, t={t10:.3f}, p={p10:.4f}")
    print(f"% positive slopes: {(slopes10 > 0).mean()*100:.1f}%")
    pb10 = res10['pb_diff'].values
    t_pb10, p_pb10 = stats.ttest_1samp(pb10, 0)
    print(f"Mean PB diff: {pb10.mean():.2f}, t={t_pb10:.3f}, p={p_pb10:.4f}")
    wu10 = res10['warmup_diff'].values
    t_wu10, p_wu10 = stats.ttest_1samp(wu10, 0)
    print(f"Warmup diff: {wu10.mean():.2f}, t={t_wu10:.3f}, p={p_wu10:.4f}")
else:
    print("No users with >=10 attempts")

print(f"\n{'='*60}")
print("DONE. All plots saved to analysis_B2/")
print(f"{'='*60}")
