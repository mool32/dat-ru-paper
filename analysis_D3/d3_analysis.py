"""
D3: Time-on-task analysis for DAT-RU
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess
import warnings
warnings.filterwarnings('ignore')

OUT = '/Users/teo/Downloads/datcreativity/analysis_D3'
plt.rcParams.update({'figure.dpi': 150, 'savefig.dpi': 150, 'font.size': 11})

# ─── 1. Load and clean ───
df = pd.read_csv('/Users/teo/Downloads/DAT-RU Results - Sheet1 (1).csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['time_spent'] = pd.to_numeric(df['time_spent'], errors='coerce')
df['score'] = pd.to_numeric(df['score'], errors='coerce')

n_total = len(df)
n_na_time = df['time_spent'].isna().sum()
n_na_score = df['score'].isna().sum()

# Remove NAs first
df_clean = df.dropna(subset=['time_spent', 'score']).copy()

# Outlier removal
n_too_fast = (df_clean['time_spent'] < 10).sum()
n_too_slow = (df_clean['time_spent'] > 1800).sum()
df_clean = df_clean[(df_clean['time_spent'] >= 10) & (df_clean['time_spent'] <= 1800)]

print("=" * 60)
print("STEP 1: Data Cleaning")
print("=" * 60)
print(f"Total records: {n_total}")
print(f"NA time_spent: {n_na_time}")
print(f"NA score: {n_na_score}")
print(f"Removed < 10s (bots/errors): {n_too_fast}")
print(f"Removed > 1800s (abandoned tabs): {n_too_slow}")
print(f"Remaining after cleaning: {len(df_clean)}")

# ─── 2. Basic statistics ───
print("\n" + "=" * 60)
print("STEP 2: Time Spent Statistics (after cleaning)")
print("=" * 60)
ts = df_clean['time_spent']
percentiles = ts.quantile([0.05, 0.25, 0.50, 0.75, 0.95])
print(f"Mean:   {ts.mean():.1f} s")
print(f"Median: {ts.median():.1f} s")
print(f"SD:     {ts.std():.1f} s")
print(f"Min:    {ts.min():.0f} s")
print(f"Max:    {ts.max():.0f} s")
print(f"Skewness: {ts.skew():.2f}")
print("Percentiles:")
for p, v in percentiles.items():
    print(f"  {p*100:.0f}th: {v:.1f} s")

# Distribution plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(ts, bins=100, edgecolor='none', alpha=0.8, color='steelblue')
axes[0].axvline(ts.median(), color='red', ls='--', label=f'Median={ts.median():.0f}s')
axes[0].axvline(ts.mean(), color='orange', ls='--', label=f'Mean={ts.mean():.0f}s')
axes[0].set_xlabel('Time Spent (seconds)')
axes[0].set_ylabel('Count')
axes[0].set_title('Distribution of Time Spent')
axes[0].legend()

axes[1].hist(ts[ts <= 600], bins=80, edgecolor='none', alpha=0.8, color='steelblue')
axes[1].axvline(ts.median(), color='red', ls='--', label=f'Median={ts.median():.0f}s')
axes[1].set_xlabel('Time Spent (seconds)')
axes[1].set_ylabel('Count')
axes[1].set_title('Distribution of Time Spent (≤10 min zoom)')
axes[1].legend()
plt.tight_layout()
plt.savefig(f'{OUT}/d3_time_distribution.png')
plt.close()

# ─── 3. Score vs Time ───
print("\n" + "=" * 60)
print("STEP 3: Score vs Time Spent")
print("=" * 60)

# 3a. Hex binning with LOWESS
fig, ax = plt.subplots(figsize=(12, 7))
hb = ax.hexbin(df_clean['time_spent'], df_clean['score'], gridsize=60,
               cmap='YlOrRd', mincnt=1, linewidths=0.2)
plt.colorbar(hb, ax=ax, label='Count')

# LOWESS
lw = lowess(df_clean['score'].values, df_clean['time_spent'].values, frac=0.15, it=3)
ax.plot(lw[:, 0], lw[:, 1], 'b-', lw=2.5, label='LOWESS smooth')
ax.set_xlabel('Time Spent (seconds)')
ax.set_ylabel('DAT Score')
ax.set_title('DAT Score vs Time Spent (hex-binned density)')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUT}/d3_score_vs_time_hex.png')
plt.close()

# 3b. Decile binning
df_clean['time_decile'] = pd.qcut(df_clean['time_spent'], 10, labels=False, duplicates='drop')
decile_stats = df_clean.groupby('time_decile').agg(
    time_mean=('time_spent', 'mean'),
    time_median=('time_spent', 'median'),
    score_mean=('score', 'mean'),
    score_median=('score', 'median'),
    score_sd=('score', 'std'),
    n=('score', 'count')
).reset_index()

print("\nDecile Analysis:")
print(decile_stats.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(decile_stats['time_mean'], decile_stats['score_mean'],
            yerr=decile_stats['score_sd'] / np.sqrt(decile_stats['n']),
            fmt='o-', capsize=5, capthick=1.5, color='steelblue', lw=2, markersize=8)
ax.set_xlabel('Mean Time Spent in Decile (seconds)')
ax.set_ylabel('Mean DAT Score')
ax.set_title('Mean DAT Score by Time Decile (±SE)')
for i, row in decile_stats.iterrows():
    ax.annotate(f'n={row["n"]:.0f}', (row['time_mean'], row['score_mean']),
                textcoords='offset points', xytext=(0, 12), fontsize=8, ha='center')
plt.tight_layout()
plt.savefig(f'{OUT}/d3_score_by_time_decile.png')
plt.close()

# 3c. Quadratic vs linear model
from numpy.polynomial import polynomial as P

x = df_clean['time_spent'].values
y = df_clean['score'].values

# Linear
slope, intercept, r_lin, p_lin, se_lin = stats.linregress(x, y)
r2_lin = r_lin**2

# Quadratic
x_mean = x.mean()
x_std = x.std()
x_z = (x - x_mean) / x_std  # standardize to avoid numerical issues

from sklearn.linear_model import LinearRegression
X_quad = np.column_stack([x_z, x_z**2])
from numpy.linalg import lstsq
X_design = np.column_stack([np.ones(len(x_z)), x_z, x_z**2])
beta, residuals, rank, sv = lstsq(X_design, y, rcond=None)
y_pred_quad = X_design @ beta
ss_res = np.sum((y - y_pred_quad)**2)
ss_tot = np.sum((y - y.mean())**2)
r2_quad = 1 - ss_res / ss_tot

# F-test for quadratic term
n = len(y)
# Linear residuals
y_pred_lin = intercept + slope * x
ss_res_lin = np.sum((y - y_pred_lin)**2)
f_stat = ((ss_res_lin - ss_res) / 1) / (ss_res / (n - 3))
p_quad_term = 1 - stats.f.cdf(f_stat, 1, n - 3)

# Find peak of quadratic
# y = beta0 + beta1*x_z + beta2*x_z^2
# dy/dx_z = beta1 + 2*beta2*x_z = 0 => x_z_peak = -beta1/(2*beta2)
x_z_peak = -beta[1] / (2 * beta[2])
x_peak = x_z_peak * x_std + x_mean

print(f"\nLinear model: R²={r2_lin:.6f}, slope={slope:.4f}, p={p_lin:.2e}")
print(f"Quadratic model: R²={r2_quad:.6f}")
print(f"Quadratic term F={f_stat:.2f}, p={p_quad_term:.2e}")
print(f"Quadratic coefficients (standardized x): b0={beta[0]:.4f}, b1={beta[1]:.4f}, b2={beta[2]:.4f}")
print(f"Peak of quadratic: {x_peak:.0f} seconds")
if beta[2] < 0:
    print("  --> Inverted U confirmed (negative quadratic term)")
else:
    print("  --> No inverted U (positive quadratic term)")

# 3d. LOWESS peak finding (exclude sparse boundaries)
lw_sorted = lw[np.argsort(lw[:, 0])]
# Only look for peak within 5th-95th percentile of time (dense region)
t5, t95 = np.percentile(x, [5, 95])
mask_dense = (lw_sorted[:, 0] >= t5) & (lw_sorted[:, 0] <= t95)
lw_dense = lw_sorted[mask_dense]
peak_idx_dense = np.argmax(lw_dense[:, 1])
print(f"\nLOWESS peak (within 5th-95th pctile): time={lw_dense[peak_idx_dense, 0]:.0f}s, score={lw_dense[peak_idx_dense, 1]:.2f}")
# Also report the score plateau range
print(f"LOWESS score range in dense region: {lw_dense[:, 1].min():.2f} - {lw_dense[:, 1].max():.2f}")
# Check if it's more of a plateau: score at 25th vs 75th pctile
t25, t75 = np.percentile(x, [25, 75])
idx25 = np.argmin(np.abs(lw_sorted[:, 0] - t25))
idx75 = np.argmin(np.abs(lw_sorted[:, 0] - t75))
print(f"LOWESS at 25th pctile ({t25:.0f}s): {lw_sorted[idx25, 1]:.2f}")
print(f"LOWESS at 75th pctile ({t75:.0f}s): {lw_sorted[idx75, 1]:.2f}")

# Plot model fits
fig, ax = plt.subplots(figsize=(10, 6))
x_plot = np.linspace(x.min(), x.max(), 500)
x_z_plot = (x_plot - x_mean) / x_std

ax.plot(x_plot, intercept + slope * x_plot, 'g--', lw=2, label=f'Linear (R²={r2_lin:.4f})')
ax.plot(x_plot, beta[0] + beta[1]*x_z_plot + beta[2]*x_z_plot**2, 'r-', lw=2,
        label=f'Quadratic (R²={r2_quad:.4f})')
ax.plot(lw_sorted[:, 0], lw_sorted[:, 1], 'b-', lw=2.5, label='LOWESS')
ax.axvline(x_peak, color='red', ls=':', alpha=0.5, label=f'Quad peak={x_peak:.0f}s')
ax.axvline(lw_dense[peak_idx_dense, 0], color='blue', ls=':', alpha=0.5,
           label=f'LOWESS peak={lw_dense[peak_idx_dense, 0]:.0f}s')
ax.set_xlabel('Time Spent (seconds)')
ax.set_ylabel('DAT Score')
ax.set_title('Model Fits: Score vs Time')
ax.legend()
ax.set_ylim(y.mean() - 2*y.std(), y.mean() + 1.5*y.std())
plt.tight_layout()
plt.savefig(f'{OUT}/d3_model_fits.png')
plt.close()

# ─── 4. Time per word ───
print("\n" + "=" * 60)
print("STEP 4: Time Per Word")
print("=" * 60)
df_clean['time_per_word'] = df_clean['time_spent'] / 7
tpw = df_clean['time_per_word']
print(f"Mean time/word: {tpw.mean():.1f}s")
print(f"Median time/word: {tpw.median():.1f}s")
print(f"5th percentile: {tpw.quantile(0.05):.1f}s")
print(f"95th percentile: {tpw.quantile(0.95):.1f}s")

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(tpw[tpw <= 60], bins=80, edgecolor='none', alpha=0.8, color='steelblue')
ax.axvline(tpw.median(), color='red', ls='--', label=f'Median={tpw.median():.1f}s')
ax.set_xlabel('Time Per Word (seconds)')
ax.set_ylabel('Count')
ax.set_title('Distribution of Time Per Word (≤60s zoom)')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUT}/d3_time_per_word.png')
plt.close()

# Score vs time_per_word binned
df_clean['tpw_bin'] = pd.qcut(df_clean['time_per_word'], 10, labels=False, duplicates='drop')
tpw_stats = df_clean.groupby('tpw_bin').agg(
    tpw_mean=('time_per_word', 'mean'),
    score_mean=('score', 'mean'),
    score_sd=('score', 'std'),
    n=('score', 'count')
).reset_index()

fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(tpw_stats['tpw_mean'], tpw_stats['score_mean'],
            yerr=tpw_stats['score_sd'] / np.sqrt(tpw_stats['n']),
            fmt='o-', capsize=5, color='steelblue', lw=2, markersize=8)
ax.set_xlabel('Mean Time Per Word (seconds)')
ax.set_ylabel('Mean DAT Score')
ax.set_title('DAT Score by Time Per Word Decile (±SE)')
plt.tight_layout()
plt.savefig(f'{OUT}/d3_score_by_time_per_word.png')
plt.close()

# ─── 5. Time of day (Moscow time) ───
print("\n" + "=" * 60)
print("STEP 5: Time of Day Analysis (Moscow Time, UTC+3)")
print("=" * 60)

df_clean['ts_moscow'] = df_clean['timestamp'] + pd.Timedelta(hours=3)
df_clean['hour'] = df_clean['ts_moscow'].dt.hour

# First-attempt users: group by user_agent, take first submission
df_sorted = df_clean.sort_values('timestamp')
df_first = df_sorted.drop_duplicates(subset='user_agent', keep='first').copy()
df_repeat = df_sorted[df_sorted.duplicated(subset='user_agent', keep='first')].copy()

n_unique_ua = df_clean['user_agent'].nunique()
print(f"Unique user_agents: {n_unique_ua}")
print(f"First-attempt records: {len(df_first)}")
print(f"Repeat records: {len(df_repeat)}")

# Hourly stats
hourly_all = df_clean.groupby('hour').agg(
    score_mean=('score', 'mean'), score_se=('score', lambda x: x.std()/np.sqrt(len(x))),
    n=('score', 'count')
).reset_index()

hourly_first = df_first.groupby('hour').agg(
    score_mean=('score', 'mean'), score_se=('score', lambda x: x.std()/np.sqrt(len(x))),
    n=('score', 'count')
).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
ax.errorbar(hourly_all['hour'], hourly_all['score_mean'], yerr=hourly_all['score_se'],
            fmt='o-', capsize=3, color='steelblue', lw=2, label='All submissions')
ax.errorbar(hourly_first['hour'], hourly_first['score_mean'], yerr=hourly_first['score_se'],
            fmt='s--', capsize=3, color='darkorange', lw=2, label='First attempt only')
ax.set_xlabel('Hour of Day (Moscow Time)')
ax.set_ylabel('Mean DAT Score')
ax.set_title('DAT Score by Hour of Day')
ax.set_xticks(range(0, 24))
ax.legend()

ax2 = axes[1]
ax2.bar(hourly_all['hour'], hourly_all['n'], alpha=0.6, color='steelblue', label='All')
ax2.bar(hourly_first['hour'], hourly_first['n'], alpha=0.6, color='darkorange', label='First attempt')
ax2.set_xlabel('Hour of Day (Moscow Time)')
ax2.set_ylabel('Number of Submissions')
ax2.set_title('Submission Volume by Hour')
ax2.set_xticks(range(0, 24))
ax2.legend()

plt.tight_layout()
plt.savefig(f'{OUT}/d3_hour_of_day.png')
plt.close()

print("\nHourly mean scores (first-attempt):")
print(hourly_first[['hour', 'score_mean', 'n']].to_string(index=False))

# ─── 6. Day of week ───
print("\n" + "=" * 60)
print("STEP 6: Day of Week Analysis")
print("=" * 60)

df_clean['dow'] = df_clean['ts_moscow'].dt.dayofweek  # 0=Mon
df_first['ts_moscow'] = df_first['timestamp'] + pd.Timedelta(hours=3)
df_first['dow'] = df_first['ts_moscow'].dt.dayofweek

dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

dow_all = df_clean.groupby('dow').agg(
    score_mean=('score', 'mean'), score_se=('score', lambda x: x.std()/np.sqrt(len(x))),
    n=('score', 'count')
).reset_index()

dow_first = df_first.groupby('dow').agg(
    score_mean=('score', 'mean'), score_se=('score', lambda x: x.std()/np.sqrt(len(x))),
    n=('score', 'count')
).reset_index()

fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(dow_all['dow'], dow_all['score_mean'], yerr=dow_all['score_se'],
            fmt='o-', capsize=5, color='steelblue', lw=2, markersize=8, label='All')
ax.errorbar(dow_first['dow'], dow_first['score_mean'], yerr=dow_first['score_se'],
            fmt='s--', capsize=5, color='darkorange', lw=2, markersize=8, label='First attempt')
ax.set_xticks(range(7))
ax.set_xticklabels(dow_names)
ax.set_xlabel('Day of Week')
ax.set_ylabel('Mean DAT Score')
ax.set_title('DAT Score by Day of Week')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUT}/d3_day_of_week.png')
plt.close()

# Weekday vs weekend
df_clean['is_weekend'] = df_clean['dow'] >= 5
wk_stats = df_clean.groupby('is_weekend')['score'].agg(['mean', 'std', 'count'])
print("\nWeekday vs Weekend (all):")
print(wk_stats)
t_wk, p_wk = stats.ttest_ind(
    df_clean[~df_clean['is_weekend']]['score'],
    df_clean[df_clean['is_weekend']]['score']
)
print(f"t={t_wk:.3f}, p={p_wk:.4f}")

print("\nDay of week scores (all):")
for _, row in dow_all.iterrows():
    print(f"  {dow_names[int(row['dow'])]}: {row['score_mean']:.2f} (n={row['n']:.0f})")

# ─── 7. Device type ───
print("\n" + "=" * 60)
print("STEP 7: Device Type Analysis")
print("=" * 60)

def categorize_device(ua):
    if pd.isna(ua):
        return 'Other'
    ua = str(ua)
    if 'iPhone' in ua or 'iPad' in ua:
        return 'iPhone/iPad'
    elif 'Android' in ua:
        return 'Android'
    elif 'Mobile' not in ua and ('Windows' in ua or 'Macintosh' in ua or 'Linux' in ua):
        return 'Desktop'
    else:
        return 'Other'

df_clean['device'] = df_clean['user_agent'].apply(categorize_device)

device_stats = df_clean.groupby('device').agg(
    n=('score', 'count'),
    score_mean=('score', 'mean'),
    score_sd=('score', 'std'),
    time_mean=('time_spent', 'mean'),
    time_median=('time_spent', 'median'),
    time_sd=('time_spent', 'std')
).reset_index()

print("\nDevice statistics:")
print(device_stats.to_string(index=False))

# ANOVA for score
device_groups = [g['score'].values for _, g in df_clean.groupby('device')]
f_dev, p_dev = stats.f_oneway(*device_groups)
print(f"\nANOVA score across devices: F={f_dev:.2f}, p={p_dev:.2e}")

# ANOVA for time
time_groups = [g['time_spent'].values for _, g in df_clean.groupby('device')]
f_time, p_time = stats.f_oneway(*time_groups)
print(f"ANOVA time across devices: F={f_time:.2f}, p={p_time:.2e}")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
order = device_stats.sort_values('n', ascending=False)['device'].tolist()

sns.boxplot(data=df_clean, x='device', y='score', order=order, ax=axes[0],
            showfliers=False, palette='Set2')
axes[0].set_title('DAT Score by Device')
axes[0].set_xlabel('Device')
axes[0].set_ylabel('DAT Score')
for i, dev in enumerate(order):
    n = device_stats[device_stats['device'] == dev]['n'].values[0]
    axes[0].text(i, axes[0].get_ylim()[1], f'n={n}', ha='center', va='bottom', fontsize=9)

sns.boxplot(data=df_clean[df_clean['time_spent'] <= 600], x='device', y='time_spent',
            order=order, ax=axes[1], showfliers=False, palette='Set2')
axes[1].set_title('Time Spent by Device (≤10 min)')
axes[1].set_xlabel('Device')
axes[1].set_ylabel('Time Spent (seconds)')

plt.tight_layout()
plt.savefig(f'{OUT}/d3_device_comparison.png')
plt.close()

# ─── 8. Temporal trend ───
print("\n" + "=" * 60)
print("STEP 8: Temporal Trends")
print("=" * 60)

df_clean['date'] = df_clean['ts_moscow'].dt.date
df_first['date'] = df_first['ts_moscow'].dt.date

daily_all = df_clean.groupby('date').agg(
    score_mean=('score', 'mean'),
    score_se=('score', lambda x: x.std()/np.sqrt(len(x))),
    n=('score', 'count'),
    time_mean=('time_spent', 'mean')
).reset_index()
daily_all['date'] = pd.to_datetime(daily_all['date'])

daily_first = df_first.groupby('date').agg(
    score_mean=('score', 'mean'),
    score_se=('score', lambda x: x.std()/np.sqrt(len(x))),
    n=('score', 'count')
).reset_index()
daily_first['date'] = pd.to_datetime(daily_first['date'])

fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True)

# Daily submissions
axes[0].bar(daily_all['date'], daily_all['n'], color='steelblue', alpha=0.7, label='All')
axes[0].bar(daily_first['date'], daily_first['n'], color='darkorange', alpha=0.7, label='First attempt')
axes[0].set_ylabel('Number of Submissions')
axes[0].set_title('Daily Submission Volume')
axes[0].legend()

# Daily mean score
axes[1].plot(daily_all['date'], daily_all['score_mean'], 'o-', color='steelblue',
             markersize=4, lw=1, label='All', alpha=0.7)
axes[1].fill_between(daily_all['date'],
                     daily_all['score_mean'] - daily_all['score_se'],
                     daily_all['score_mean'] + daily_all['score_se'],
                     alpha=0.2, color='steelblue')
if len(daily_first) > 3:
    axes[1].plot(daily_first['date'], daily_first['score_mean'], 's--', color='darkorange',
                 markersize=4, lw=1, label='First attempt', alpha=0.7)
    axes[1].fill_between(daily_first['date'],
                         daily_first['score_mean'] - daily_first['score_se'],
                         daily_first['score_mean'] + daily_first['score_se'],
                         alpha=0.2, color='darkorange')
axes[1].set_ylabel('Mean DAT Score')
axes[1].set_title('Daily Mean Score Over Collection Period')
axes[1].legend()

# Daily mean time
axes[2].plot(daily_all['date'], daily_all['time_mean'], 'o-', color='steelblue',
             markersize=4, lw=1, alpha=0.7)
axes[2].set_ylabel('Mean Time Spent (s)')
axes[2].set_xlabel('Date')
axes[2].set_title('Daily Mean Time Spent')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(f'{OUT}/d3_temporal_trends.png')
plt.close()

# Trend test (Spearman correlation of date vs score for first-attempt)
df_first_sorted = df_first.sort_values('timestamp')
day_num = (df_first_sorted['timestamp'] - df_first_sorted['timestamp'].min()).dt.total_seconds()
rho_score, p_score = stats.spearmanr(day_num, df_first_sorted['score'])
rho_time, p_time_trend = stats.spearmanr(day_num, df_first_sorted['time_spent'])

print(f"\nFirst-attempt trends (Spearman):")
print(f"  Score vs day: rho={rho_score:.4f}, p={p_score:.4f}")
print(f"  Time vs day:  rho={rho_time:.4f}, p={p_time_trend:.4f}")

# Collection period
date_range = df_clean['timestamp'].max() - df_clean['timestamp'].min()
print(f"\nCollection period: {date_range.days} days")
print(f"  From: {df_clean['timestamp'].min()}")
print(f"  To:   {df_clean['timestamp'].max()}")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE. All plots saved to:")
print(f"  {OUT}/")
print("=" * 60)
