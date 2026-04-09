"""
E5: Optimal Stopping Analysis — The Cost of Persistence
DAT-RU dataset
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.special import ndtri  # inverse normal CDF = Phi^{-1}
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

OUT = Path("/Users/teo/Downloads/datcreativity/analysis_E5")
OUT.mkdir(parents=True, exist_ok=True)

# ── 1. Load & classify ─────────────────────────────────────────────────
df = pd.read_csv("/Users/teo/Downloads/DAT-RU Results - Sheet1 (1).csv")
df.columns = ['timestamp', 'score', 'words', 'time_spent', 'user_agent']
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

# Classify specific vs generic UAs
import re

def is_specific_ua(ua):
    """Specific UAs contain device model identifiers."""
    if pd.isna(ua):
        return False
    # iPhone model: iPhone17,1 etc
    if re.search(r'iPhone\d+,\d+', ua):
        return True
    # Android device model: SM-S928B, Pixel 9, 2409BRN2CA, etc.
    # Look for Build/ which typically contains model info
    if re.search(r';\s*[A-Z0-9][\w\-\.]+\s+Build/', ua):
        return True
    # iPad model
    if re.search(r'iPad\d+,\d+', ua):
        return True
    return False

df['specific_ua'] = df['user_agent'].apply(is_specific_ua)

# Attempt numbering per user_agent
df['attempt'] = df.groupby('user_agent').cumcount() + 1
ua_counts = df.groupby('user_agent').size().reset_index(name='total_attempts')

# Groups
specific_uas_10 = set(ua_counts[(ua_counts['total_attempts'] >= 10)]['user_agent']) & \
                  set(df[df['specific_ua']]['user_agent'])
all_uas_10 = set(ua_counts[ua_counts['total_attempts'] >= 10]['user_agent'])

print(f"Total records: {len(df)}")
print(f"Unique UAs: {df['user_agent'].nunique()}")
print(f"Specific UAs with >=10 attempts: {len(specific_uas_10)}")
print(f"All UAs with >=10 attempts: {len(all_uas_10)}")

# ── 2. Personal best curves & i.i.d. theory ────────────────────────────

def expected_max_normal(n):
    """E[max(Z_1,...,Z_n)] for Z_i ~ N(0,1), computed numerically."""
    if n <= 0:
        return 0.0
    # Use order statistic formula: E[Z_(n)] = integral from -inf to inf of z * n * phi(z) * Phi(z)^(n-1) dz
    # Or approximation: Phi^{-1}(n/(n+1))
    # For better accuracy, use numerical integration for small n, approximation for large n
    if n <= 200:
        from scipy.integrate import quad
        def integrand(z):
            return z * n * stats.norm.pdf(z) * stats.norm.cdf(z)**(n-1)
        val, _ = quad(integrand, -6, 6)
        return val
    else:
        return ndtri(n / (n + 1))

# Precompute expected max for n=1..300
print("Precomputing expected max order statistics...")
max_n = 300
E_max_z = np.zeros(max_n + 1)
for n in range(1, max_n + 1):
    E_max_z[n] = expected_max_normal(n)
print(f"  E[max(1)]={E_max_z[1]:.4f}, E[max(5)]={E_max_z[5]:.4f}, E[max(10)]={E_max_z[10]:.4f}, E[max(50)]={E_max_z[50]:.4f}")

def compute_user_stats(user_df):
    """Compute personal best curve and stats for one user."""
    scores = user_df.sort_values('attempt')['score'].values
    times = user_df.sort_values('attempt')['time_spent'].values
    N = len(scores)

    pb = np.maximum.accumulate(scores)
    mu = np.mean(scores)
    sigma = np.std(scores, ddof=1) if N > 1 else 0.0

    # Theoretical PB under i.i.d.
    theo_pb = np.array([mu + sigma * E_max_z[min(n, max_n)] for n in range(1, N+1)])

    return {
        'scores': scores,
        'times': times,
        'pb': pb,
        'theo_pb': theo_pb,
        'mu': mu,
        'sigma': sigma,
        'N': N
    }

# Compute for both groups
print("\nComputing user stats for specific UAs...")
specific_stats = {}
for ua in specific_uas_10:
    user_df = df[df['user_agent'] == ua]
    specific_stats[ua] = compute_user_stats(user_df)

print(f"  Done: {len(specific_stats)} users")

print("Computing user stats for all UAs...")
all_stats = {}
for ua in all_uas_10:
    user_df = df[df['user_agent'] == ua]
    all_stats[ua] = compute_user_stats(user_df)

print(f"  Done: {len(all_stats)} users")

# ── 3. Marginal value of next attempt ──────────────────────────────────

def compute_marginal_gains(stats_dict, max_attempt=100):
    """Compute average ΔPB(n) across users, both real and theoretical."""
    real_gains = {n: [] for n in range(2, max_attempt+1)}
    theo_gains = {n: [] for n in range(2, max_attempt+1)}

    for ua, s in stats_dict.items():
        N = s['N']
        for n in range(2, min(N+1, max_attempt+1)):
            real_gains[n].append(s['pb'][n-1] - s['pb'][n-2])
            theo_gains[n].append(s['theo_pb'][n-1] - s['theo_pb'][n-2])

    attempts = []
    mean_real = []
    mean_theo = []
    se_real = []
    n_users = []

    for n in range(2, max_attempt+1):
        if len(real_gains[n]) >= 5:
            attempts.append(n)
            mean_real.append(np.mean(real_gains[n]))
            mean_theo.append(np.mean(theo_gains[n]))
            se_real.append(np.std(real_gains[n]) / np.sqrt(len(real_gains[n])))
            n_users.append(len(real_gains[n]))

    return np.array(attempts), np.array(mean_real), np.array(mean_theo), np.array(se_real), np.array(n_users)

# Compute for specific UAs
att_s, real_s, theo_s, se_s, nu_s = compute_marginal_gains(specific_stats)
att_a, real_a, theo_a, se_a, nu_a = compute_marginal_gains(all_stats)

# Plot 1: Marginal gains
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, att, real, theo, se, nu, title in [
    (axes[0], att_s, real_s, theo_s, se_s, nu_s, "Specific UAs"),
    (axes[1], att_a, real_a, theo_a, se_a, nu_a, "All UAs")
]:
    mask = att <= 80
    ax.fill_between(att[mask], (real - 1.96*se)[mask], (real + 1.96*se)[mask], alpha=0.2, color='C0')
    ax.plot(att[mask], real[mask], 'C0-', lw=2, label='Observed ΔPB')
    ax.plot(att[mask], theo[mask], 'C1--', lw=2, label='i.i.d. predicted ΔPB')
    ax.axhline(0.5, color='gray', ls=':', lw=1, label='ΔPB = 0.5 threshold')
    ax.axhline(0, color='black', ls='-', lw=0.5)
    ax.set_xlabel('Attempt number n')
    ax.set_ylabel('Mean ΔPB(n) = PB(n) − PB(n−1)')
    ax.set_title(f'Marginal Value of Next Attempt — {title}')
    ax.legend(fontsize=9)
    ax.set_xlim(2, 80)

    # Add user count on secondary axis
    ax2 = ax.twinx()
    ax2.bar(att[mask], nu[mask], alpha=0.1, color='gray', width=0.8)
    ax2.set_ylabel('N users at this attempt', color='gray', alpha=0.5)
    ax2.tick_params(axis='y', labelcolor='gray')

plt.tight_layout()
plt.savefig(OUT / "E5_01_marginal_gains.png", dpi=150, bbox_inches='tight')
plt.close()
print("\nPlot 1 saved: E5_01_marginal_gains.png")

# ── 4. Cost of persistence ─────────────────────────────────────────────

# For users with >=50 attempts
def persistence_analysis(stats_dict, label):
    """Analyze PB at milestones and time investment."""
    heavy_users = {ua: s for ua, s in stats_dict.items() if s['N'] >= 50}
    print(f"\n--- Cost of Persistence: {label} ---")
    print(f"Users with >=50 attempts: {len(heavy_users)}")

    if len(heavy_users) < 3:
        print("  Too few users for analysis")
        return None

    pb_at_10 = [s['pb'][9] for s in heavy_users.values()]
    pb_at_50 = [s['pb'][49] for s in heavy_users.values()]
    pb_at_final = [s['pb'][-1] for s in heavy_users.values()]

    mean_scores = [s['mu'] for s in heavy_users.values()]

    # Time investment
    time_at_10 = [np.sum(s['times'][:10]) for s in heavy_users.values()]
    time_at_50 = [np.sum(s['times'][:50]) for s in heavy_users.values()]
    time_at_final = [np.sum(s['times']) for s in heavy_users.values()]

    total_attempts = [s['N'] for s in heavy_users.values()]

    print(f"  PB at attempt 10: {np.mean(pb_at_10):.2f} ± {np.std(pb_at_10):.2f}")
    print(f"  PB at attempt 50: {np.mean(pb_at_50):.2f} ± {np.std(pb_at_50):.2f}")
    print(f"  PB at final:      {np.mean(pb_at_final):.2f} ± {np.std(pb_at_final):.2f}")
    print(f"  Gain 11→50:       {np.mean(np.array(pb_at_50)-np.array(pb_at_10)):.2f} pts")
    print(f"  Gain 51→final:    {np.mean(np.array(pb_at_final)-np.array(pb_at_50)):.2f} pts")
    print(f"  Time at 10:       {np.mean(time_at_10)/60:.1f} min")
    print(f"  Time at 50:       {np.mean(time_at_50)/60:.1f} min")
    print(f"  Time at final:    {np.mean(time_at_final)/60:.1f} min")
    print(f"  Mean total attempts: {np.mean(total_attempts):.0f}")

    return {
        'pb_10': pb_at_10, 'pb_50': pb_at_50, 'pb_final': pb_at_final,
        'time_10': time_at_10, 'time_50': time_at_50, 'time_final': time_at_final,
        'mean_scores': mean_scores, 'total_attempts': total_attempts,
        'heavy_users': heavy_users
    }

persist_specific = persistence_analysis(specific_stats, "Specific UAs")
persist_all = persistence_analysis(all_stats, "All UAs")

# Plot 2: Diminishing returns — time invested vs PB
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, stats_dict, title in [
    (axes[0], specific_stats, "Specific UAs"),
    (axes[1], all_stats, "All UAs")
]:
    # For each user, compute cumulative time and PB at each attempt
    # Average across users at each attempt
    max_att = 100
    cum_times = {n: [] for n in range(1, max_att+1)}
    pbs = {n: [] for n in range(1, max_att+1)}

    for ua, s in stats_dict.items():
        cum_t = np.cumsum(s['times'])
        for n in range(1, min(s['N']+1, max_att+1)):
            cum_times[n].append(cum_t[n-1])
            pbs[n].append(s['pb'][n-1])

    attempts = []
    mean_time = []
    mean_pb = []

    for n in range(1, max_att+1):
        if len(cum_times[n]) >= 5:
            attempts.append(n)
            mean_time.append(np.mean(cum_times[n]) / 60)  # minutes
            mean_pb.append(np.mean(pbs[n]))

    attempts = np.array(attempts)
    mean_time = np.array(mean_time)
    mean_pb = np.array(mean_pb)

    ax.plot(mean_time, mean_pb, 'C0-', lw=2)

    # Mark milestones
    for milestone, color, marker in [(5, 'C2', 's'), (10, 'C1', 'o'), (20, 'C3', 'D'), (50, 'C4', '^')]:
        idx = np.argmin(np.abs(attempts - milestone))
        if attempts[idx] == milestone:
            ax.plot(mean_time[idx], mean_pb[idx], marker, color=color, ms=10, zorder=5,
                   label=f'Attempt {milestone}: {mean_pb[idx]:.1f} pts @ {mean_time[idx]:.1f} min')

    ax.set_xlabel('Cumulative time invested (minutes)')
    ax.set_ylabel('Mean personal best score')
    ax.set_title(f'Diminishing Returns — {title}')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT / "E5_02_diminishing_returns_time.png", dpi=150, bbox_inches='tight')
plt.close()
print("Plot 2 saved: E5_02_diminishing_returns_time.png")

# Plot 3: Points per minute at each stage
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, stats_dict, title in [
    (axes[0], specific_stats, "Specific UAs"),
    (axes[1], all_stats, "All UAs")
]:
    # Marginal points per minute for each attempt
    max_att = 80
    marginal_pts_per_min = {n: [] for n in range(2, max_att+1)}

    for ua, s in stats_dict.items():
        N = s['N']
        for n in range(2, min(N+1, max_att+1)):
            delta_pb = s['pb'][n-1] - s['pb'][n-2]
            time_n = s['times'][n-1] / 60  # minutes
            if time_n > 0:
                marginal_pts_per_min[n].append(delta_pb / time_n)

    attempts = []
    mean_ppm = []

    for n in range(2, max_att+1):
        if len(marginal_pts_per_min[n]) >= 5:
            attempts.append(n)
            mean_ppm.append(np.mean(marginal_pts_per_min[n]))

    ax.plot(attempts, mean_ppm, 'C0-', lw=2)
    ax.axhline(0, color='black', ls='-', lw=0.5)
    ax.set_xlabel('Attempt number')
    ax.set_ylabel('Mean marginal PB gain per minute')
    ax.set_title(f'Efficiency: Points Gained per Minute — {title}')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT / "E5_03_points_per_minute.png", dpi=150, bbox_inches='tight')
plt.close()
print("Plot 3 saved: E5_03_points_per_minute.png")

# ── 5. When to stop recommendation ─────────────────────────────────────

print("\n--- When to Stop Analysis ---")

# Compute typical mu and sigma across users
all_mus = [s['mu'] for s in all_stats.values()]
all_sigmas = [s['sigma'] for s in all_stats.values()]
spec_mus = [s['mu'] for s in specific_stats.values()]
spec_sigmas = [s['sigma'] for s in specific_stats.values()]

print(f"Specific UAs: median μ={np.median(spec_mus):.2f}, median σ={np.median(spec_sigmas):.2f}")
print(f"All UAs:      median μ={np.median(all_mus):.2f}, median σ={np.median(all_sigmas):.2f}")

# Under i.i.d. with typical sigma, expected gain from attempt n to n+1
typical_sigma_spec = np.median(spec_sigmas)
typical_sigma_all = np.median(all_sigmas)

print(f"\nTheoretical ΔPB under i.i.d. (typical σ):")
for label, sigma in [("Specific", typical_sigma_spec), ("All", typical_sigma_all)]:
    print(f"\n  {label} UAs (σ={sigma:.2f}):")
    for n in [1, 2, 3, 4, 5, 10, 20, 50]:
        if n < max_n:
            delta = sigma * (E_max_z[n+1] - E_max_z[n])
            print(f"    n={n:3d} → n+1: ΔPB = {delta:.3f} pts")

# Find stopping point for threshold
for label, sigma in [("Specific", typical_sigma_spec), ("All", typical_sigma_all)]:
    for threshold in [0.5, 0.25, 0.1]:
        for n in range(1, max_n):
            delta = sigma * (E_max_z[n+1] - E_max_z[n])
            if delta < threshold:
                print(f"  {label}: ΔPB < {threshold} at attempt n={n} (ΔPB={delta:.3f})")
                break

# Compare actual stopping vs optimal
print("\n--- Actual vs Optimal Stopping ---")
for label, stats_dict in [("Specific", specific_stats), ("All", all_stats)]:
    actual_stops = [s['N'] for s in stats_dict.values()]
    print(f"  {label} UAs: median actual stop = {np.median(actual_stops):.0f}, "
          f"mean = {np.mean(actual_stops):.1f}, max = {np.max(actual_stops)}")

# ── 6. Individual variation ─────────────────────────────────────────────

# Plot 4: σ vs (PB - mean) — how much volatile vs consistent performers gain
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, stats_dict, title in [
    (axes[0], specific_stats, "Specific UAs"),
    (axes[1], all_stats, "All UAs")
]:
    sigmas = np.array([s['sigma'] for s in stats_dict.values()])
    pb_gains = np.array([s['pb'][-1] - s['mu'] for s in stats_dict.values()])
    Ns = np.array([s['N'] for s in stats_dict.values()])

    # Theoretical: PB - mu = sigma * E[max(Z_1..Z_N)]
    theo_gains = np.array([s['sigma'] * E_max_z[min(s['N'], max_n)] for s in stats_dict.values()])

    sc = ax.scatter(sigmas, pb_gains, c=Ns, cmap='viridis', alpha=0.6, s=20,
                    edgecolors='none', label='Observed')
    # Plot theoretical line for different N
    sigma_range = np.linspace(0, np.percentile(sigmas, 99), 100)
    for n_ref, ls, clr in [(10, '--', 'C1'), (30, ':', 'C3'), (100, '-.', 'C4')]:
        ax.plot(sigma_range, sigma_range * E_max_z[n_ref], ls, color=clr, lw=2,
               label=f'i.i.d. theory (N={n_ref})')

    plt.colorbar(sc, ax=ax, label='N attempts')
    ax.set_xlabel('User σ (score volatility)')
    ax.set_ylabel('PB − mean (gain from best attempt)')
    ax.set_title(f'Volatility vs Best-Attempt Gain — {title}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT / "E5_04_sigma_vs_gain.png", dpi=150, bbox_inches='tight')
plt.close()
print("\nPlot 4 saved: E5_04_sigma_vs_gain.png")

# Plot 5: Who benefits from persistence?
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, stats_dict, title in [
    (axes[0], specific_stats, "Specific UAs"),
    (axes[1], all_stats, "All UAs")
]:
    # For users with enough attempts, compare PB at attempt 5 vs final PB
    min_att = 20
    users = {ua: s for ua, s in stats_dict.items() if s['N'] >= min_att}

    sigmas = np.array([s['sigma'] for s in users.values()])
    pb5 = np.array([s['pb'][4] for s in users.values()])
    pb_final = np.array([s['pb'][-1] for s in users.values()])
    extra_gain = pb_final - pb5

    ax.scatter(sigmas, extra_gain, alpha=0.5, s=20, edgecolors='none')

    # Bin and show trend
    bins = np.percentile(sigmas, np.linspace(0, 100, 11))
    bin_centers = []
    bin_means = []
    for i in range(len(bins)-1):
        mask = (sigmas >= bins[i]) & (sigmas < bins[i+1])
        if mask.sum() >= 3:
            bin_centers.append(np.mean(sigmas[mask]))
            bin_means.append(np.mean(extra_gain[mask]))

    ax.plot(bin_centers, bin_means, 'r-o', lw=2, ms=8, label='Binned mean')
    ax.axhline(0, color='black', ls='-', lw=0.5)
    ax.set_xlabel('User σ (score volatility)')
    ax.set_ylabel('PB(final) − PB(5) [extra points from persistence]')
    ax.set_title(f'Who Benefits from Persistence? (N≥{min_att}) — {title}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT / "E5_05_persistence_benefit_by_sigma.png", dpi=150, bbox_inches='tight')
plt.close()
print("Plot 5 saved: E5_05_persistence_benefit_by_sigma.png")

# ── 7. Summary figure: The Cost of Persistence ─────────────────────────

fig, ax1 = plt.subplots(figsize=(12, 7))

# Use all_stats for the main figure (larger sample)
max_att_plot = 150
attempts_range = np.arange(1, max_att_plot+1)

# Actual mean PB curve
actual_pbs = {n: [] for n in attempts_range}
for ua, s in all_stats.items():
    for n in range(1, min(s['N']+1, max_att_plot+1)):
        actual_pbs[n].append(s['pb'][n-1])

actual_x = []
actual_y = []
actual_n = []
for n in attempts_range:
    if len(actual_pbs[n]) >= 5:
        actual_x.append(n)
        actual_y.append(np.mean(actual_pbs[n]))
        actual_n.append(len(actual_pbs[n]))

actual_x = np.array(actual_x)
actual_y = np.array(actual_y)
actual_n = np.array(actual_n)

# Theoretical i.i.d. curve using median mu and sigma from all users
median_mu = np.median(all_mus)
median_sigma = np.median(all_sigmas)
theo_x = np.arange(1, max_att_plot+1)
theo_y = np.array([median_mu + median_sigma * E_max_z[n] for n in theo_x])

# Theoretical ceiling (asymptote) — use n=1000 approximation
ceiling = median_mu + median_sigma * ndtri(1000/1001)

# Plot main curves
ax1.plot(actual_x, actual_y, 'C0-', lw=2.5, label='Observed mean PB', zorder=3)
ax1.plot(theo_x, theo_y, 'C1--', lw=2, label=f'i.i.d. theory (μ={median_mu:.1f}, σ={median_sigma:.1f})', zorder=3)
ax1.axhline(ceiling, color='gray', ls=':', lw=1, alpha=0.7, label=f'Theoretical ceiling ≈ {ceiling:.1f}')

# Annotations: mark where the theoretical curve captures X% of the total possible gain
# Total possible gain = ceiling - first_attempt = sigma * (Phi^{-1}(1000/1001) - 0)
# E_max_z[1] = 0, so total_gain_z = ndtri(1000/1001) ≈ 3.09
total_gain_z = ndtri(1000/1001)
for pct_frac, pct_label, color in [(0.50, '50%', 'C2'), (0.75, '75%', 'C3'), (0.90, '90%', 'C4')]:
    target_z = pct_frac * total_gain_z  # need E_max_z[n] >= target_z
    # Find n where E_max_z[n] >= target_z
    n_cross = None
    for n in range(1, max_n+1):
        if E_max_z[n] >= target_z:
            n_cross = n
            break
    if n_cross is not None:
        target_score = median_mu + median_sigma * target_z
        ax1.axvline(n_cross, color=color, ls=':', lw=1.5, alpha=0.7)
        ax1.annotate(f'{pct_label} of max gain\nat attempt {n_cross}',
                    xy=(n_cross, target_score), xytext=(n_cross * 2.5, target_score - 1.0),
                    fontsize=9, color=color, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

ax1.set_xscale('log')
ax1.set_xlabel('Number of attempts (log scale)', fontsize=12)
ax1.set_ylabel('Expected personal best score', fontsize=12)
ax1.set_title('The Cost of Persistence', fontsize=14, fontweight='bold')

# Custom x-ticks for log scale
ax1.set_xticks([1, 2, 3, 5, 10, 20, 50, 100])
ax1.xaxis.set_major_formatter(mticker.ScalarFormatter())

# Inset: marginal gain per attempt
ax_inset = ax1.inset_axes([0.45, 0.35, 0.50, 0.35])

# Theoretical marginal gain
delta_theo = np.array([median_sigma * (E_max_z[n+1] - E_max_z[n]) for n in range(1, 100)])
delta_x = np.arange(2, 101)

# Observed marginal gain (from att_a computed earlier)
mask_plot = att_a <= 100
ax_inset.plot(att_a[mask_plot], real_a[mask_plot], 'C0-', lw=1.5, label='Observed', alpha=0.7)
ax_inset.plot(delta_x, delta_theo, 'C1--', lw=1.5, label='i.i.d. theory')
ax_inset.axhline(0.5, color='gray', ls=':', lw=1)
ax_inset.axhline(0, color='black', ls='-', lw=0.5)
ax_inset.set_xlabel('Attempt n', fontsize=8)
ax_inset.set_ylabel('ΔPB(n)', fontsize=8)
ax_inset.set_title('Marginal gain per attempt', fontsize=9)
ax_inset.legend(fontsize=7)
ax_inset.tick_params(labelsize=7)
ax_inset.set_xlim(2, 60)
ax_inset.grid(True, alpha=0.2)

# Main legend
ax1.legend(fontsize=10, loc='upper left')
ax1.grid(True, alpha=0.3)

# Add sample size annotation
ax1.text(0.98, 0.02, f'N = {len(all_stats)} users with ≥10 attempts',
        transform=ax1.transAxes, ha='right', va='bottom', fontsize=9, color='gray')

plt.savefig(OUT / "E5_07_summary_cost_of_persistence.png", dpi=200, bbox_inches='tight')
plt.close()
print("\nPlot 7 saved: E5_07_summary_cost_of_persistence.png")

# ── Save numerical summary ──────────────────────────────────────────────

print("\n" + "="*70)
print("NUMERICAL SUMMARY")
print("="*70)

# Compile key numbers
print(f"\n1. SAMPLE: {len(specific_stats)} specific-UA users, {len(all_stats)} all-UA users (≥10 attempts)")
print(f"   Median μ: {median_mu:.2f}, Median σ: {median_sigma:.2f}")
print(f"   Theoretical ceiling: {ceiling:.2f}")

# X% of max gain annotations (use approximation for large n: E[max(n)] ~ Phi^{-1}(n/(n+1)))
total_gain_z_summary = ndtri(1000/1001)
for pct_frac, pct_label in [(0.50, '50%'), (0.75, '75%'), (0.90, '90%'), (0.95, '95%'), (0.99, '99%')]:
    target_z = pct_frac * total_gain_z_summary
    # For n <= max_n use precomputed; for larger n use approximation
    found = False
    for n in range(1, max_n+1):
        if E_max_z[n] >= target_z:
            print(f"   {pct_label} of max gain reached at attempt {n}")
            found = True
            break
    if not found:
        # Use Phi^{-1}(n/(n+1)) >= target_z => n/(n+1) >= Phi(target_z) => n >= Phi(target_z)/(1-Phi(target_z))
        p = stats.norm.cdf(target_z)
        n_approx = int(np.ceil(p / (1 - p)))
        print(f"   {pct_label} of max gain reached at attempt ~{n_approx}")

print(f"\n2. MARGINAL GAINS (all UAs):")
for n_check in [2, 3, 5, 10, 20, 50]:
    idx = np.where(att_a == n_check)[0]
    if len(idx) > 0:
        i = idx[0]
        print(f"   Attempt {n_check:3d}: observed ΔPB = {real_a[i]:.3f}, theoretical = {theo_a[i]:.3f}")

# Cost of persistence for heavy users
if persist_all is not None:
    print(f"\n3. COST OF PERSISTENCE (users with ≥50 attempts, all UAs):")
    pb10 = np.mean(persist_all['pb_10'])
    pb50 = np.mean(persist_all['pb_50'])
    pbfin = np.mean(persist_all['pb_final'])
    t10 = np.mean(persist_all['time_10']) / 60
    t50 = np.mean(persist_all['time_50']) / 60
    tfin = np.mean(persist_all['time_final']) / 60
    print(f"   PB@10:  {pb10:.2f} pts, time invested: {t10:.1f} min")
    print(f"   PB@50:  {pb50:.2f} pts, time invested: {t50:.1f} min")
    print(f"   PB@end: {pbfin:.2f} pts, time invested: {tfin:.1f} min")
    print(f"   Gain 11→50: +{pb50-pb10:.2f} pts for +{t50-t10:.1f} min ({(pb50-pb10)/(t50-t10)*60:.2f} pts/hr)")
    print(f"   Gain 51→end: +{pbfin-pb50:.2f} pts for +{tfin-t50:.1f} min ({(pbfin-pb50)/(tfin-t50)*60:.2f} pts/hr)")
    print(f"   First 10: {(pb10 - median_mu)/(t10)*60:.2f} pts/hr above mean")

print(f"\n4. STOPPING RECOMMENDATION:")
for threshold in [0.5, 0.25, 0.1]:
    for n in range(1, max_n):
        delta = median_sigma * (E_max_z[n+1] - E_max_z[n])
        if delta < threshold:
            print(f"   Stop when ΔPB < {threshold}: after attempt {n} (expected gain = {delta:.3f})")
            break

print("\nDone! All plots saved to:", OUT)
