"""
Analysis A2: Check whether the nonlinear adjustment function distorts key analyses.
"""

import json
import struct
import numpy as np
import pandas as pd
from scipy import stats
from wordfreq import word_frequency
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

OUT = Path('/Users/teo/Downloads/datcreativity/analysis_A2')
DATA = Path('/Users/teo/Downloads/datcreativity/data')

# ── 0. Load data ─────────────────────────────────────────────────────────

df = pd.read_csv('/Users/teo/Downloads/DAT-RU Results - Sheet1 (1).csv')
print(f"Loaded {len(df)} records")
print(f"Score range: {df['score'].min():.2f} – {df['score'].max():.2f}")

# ── 1. Inverse function: adjusted → raw ──────────────────────────────────

def adjusted_to_raw(adj):
    """Inverse of: adjusted = 47.4548 * (raw/100)^3.3820 + 48.5157"""
    inner = (adj - 48.5157) / 47.4548
    # clamp to avoid negative values under the power
    inner = np.clip(inner, 0, None)
    raw = 100 * inner ** (1.0 / 3.3820)
    return raw

def raw_to_adjusted(raw):
    return 47.4548 * (raw / 100) ** 3.3820 + 48.5157

df['raw_score'] = adjusted_to_raw(df['score'].values)
print(f"Raw score range: {df['raw_score'].min():.4f} – {df['raw_score'].max():.4f}")

# Verify round-trip
check = raw_to_adjusted(df['raw_score'].values)
max_err = np.max(np.abs(check - df['score'].values))
print(f"Round-trip max error: {max_err:.2e}")

# ── 1b. Verify raw against embeddings for ~100 records ────────────────────

print("\n=== Embedding verification ===")

# Load embedding data
with open(DATA / 'words.json', 'r') as f:
    word_list = json.load(f)
word_to_idx = {w: i for i, w in enumerate(word_list)}
n_words = len(word_list)
print(f"Vocabulary: {n_words} words")

# Load matrix
with open(DATA / 'matrix.bin', 'rb') as f:
    header = f.read(8)
    num_w, dim = struct.unpack('II', header)
    print(f"Matrix: {num_w} x {dim}")
    raw_data = np.frombuffer(f.read(), dtype=np.int8)
    matrix = raw_data.reshape(num_w, dim).astype(np.float32)

def compute_raw_from_words(words_str):
    """Parse words, look up embeddings, compute mean pairwise cosine distance."""
    words = [w.strip().lower() for w in words_str.split(',')]
    if len(words) != 7:
        return np.nan

    vecs = []
    for w in words:
        if w in word_to_idx:
            vecs.append(matrix[word_to_idx[w]])
        else:
            return np.nan

    vecs = np.array(vecs)
    # Normalize
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    vecs_n = vecs / norms

    # Mean pairwise cosine distance = 1 - cosine_similarity
    dists = []
    for i in range(7):
        for j in range(i+1, 7):
            cos_sim = np.dot(vecs_n[i], vecs_n[j])
            dists.append(1.0 - cos_sim)

    mean_dist = np.mean(dists)
    raw = mean_dist * 100  # raw = mean cosine distance * 100
    return raw

# Sample 200 records, try to verify
np.random.seed(42)
sample_idx = np.random.choice(len(df), size=200, replace=False)
sample = df.iloc[sample_idx].copy()

verified = []
for _, row in sample.iterrows():
    raw_emb = compute_raw_from_words(row['words'])
    if not np.isnan(raw_emb):
        verified.append({
            'raw_from_inverse': row['raw_score'],
            'raw_from_embeddings': raw_emb,
            'adjusted': row['score']
        })

vdf = pd.DataFrame(verified)
print(f"Successfully verified {len(vdf)} / 200 records")
corr_raw = np.corrcoef(vdf['raw_from_inverse'], vdf['raw_from_embeddings'])[0, 1]
mean_diff = (vdf['raw_from_inverse'] - vdf['raw_from_embeddings']).mean()
std_diff = (vdf['raw_from_inverse'] - vdf['raw_from_embeddings']).std()
print(f"Correlation (inverse vs embedding): r = {corr_raw:.6f}")
print(f"Mean difference: {mean_diff:.4f}, SD: {std_diff:.4f}")

# ── 2a. Test-retest ──────────────────────────────────────────────────────

print("\n=== Test-retest analysis ===")

# Group by user_agent for repeat users
ua_counts = df.groupby('user_agent').size()
repeat_uas = ua_counts[ua_counts >= 2].index
print(f"User agents with >=2 attempts: {len(repeat_uas)}")

# For each repeat UA, take first two attempts
pairs_adj = []
pairs_raw = []
for ua in repeat_uas:
    sub = df[df['user_agent'] == ua].sort_values('timestamp').head(2)
    if len(sub) == 2:
        pairs_adj.append((sub['score'].iloc[0], sub['score'].iloc[1]))
        pairs_raw.append((sub['raw_score'].iloc[0], sub['raw_score'].iloc[1]))

pairs_adj = np.array(pairs_adj)
pairs_raw = np.array(pairs_raw)
print(f"Valid pairs: {len(pairs_adj)}")

if len(pairs_adj) > 10:
    r_adj_retest = np.corrcoef(pairs_adj[:, 0], pairs_adj[:, 1])[0, 1]
    r_raw_retest = np.corrcoef(pairs_raw[:, 0], pairs_raw[:, 1])[0, 1]
    print(f"Test-retest r (adjusted): {r_adj_retest:.4f}")
    print(f"Test-retest r (raw):      {r_raw_retest:.4f}")
else:
    r_adj_retest = r_raw_retest = np.nan
    print("Not enough pairs for test-retest")

# ── 2b. Word frequency effect ────────────────────────────────────────────

print("\n=== Word frequency analysis ===")

def mean_word_freq(words_str):
    words = [w.strip().lower() for w in words_str.split(',')]
    freqs = [word_frequency(w, 'ru') for w in words]
    return np.mean(freqs)

# Compute for all records
df['mean_freq'] = df['words'].apply(mean_word_freq)
# Log transform for better correlation
df['log_mean_freq'] = np.log10(df['mean_freq'].clip(lower=1e-10))

r_adj_freq, p_adj_freq = stats.pearsonr(df['log_mean_freq'], df['score'])
r_raw_freq, p_raw_freq = stats.pearsonr(df['log_mean_freq'], df['raw_score'])
print(f"Freq vs adjusted: r = {r_adj_freq:.4f}, p = {p_adj_freq:.2e}")
print(f"Freq vs raw:      r = {r_raw_freq:.4f}, p = {p_raw_freq:.2e}")

# ── 2c. Time-on-task ─────────────────────────────────────────────────────

print("\n=== Time-on-task analysis ===")

# Filter extreme time_spent values
df_time = df[(df['time_spent'] > 0) & (df['time_spent'] < df['time_spent'].quantile(0.99))].copy()
df_time['time_decile'] = pd.qcut(df_time['time_spent'], 10, labels=False, duplicates='drop')

time_stats = df_time.groupby('time_decile').agg(
    mean_time=('time_spent', 'mean'),
    mean_adj=('score', 'mean'),
    mean_raw=('raw_score', 'mean'),
    n=('score', 'count')
).reset_index()

r_adj_time, p_adj_time = stats.pearsonr(df_time['time_spent'], df_time['score'])
r_raw_time, p_raw_time = stats.pearsonr(df_time['time_spent'], df_time['raw_score'])
print(f"Time vs adjusted: r = {r_adj_time:.4f}, p = {p_adj_time:.2e}")
print(f"Time vs raw:      r = {r_raw_time:.4f}, p = {p_raw_time:.2e}")

# ── 3. PLOTS ──────────────────────────────────────────────────────────────

# Plot 1: Adjustment function
fig, ax = plt.subplots(figsize=(8, 6))
raw_range = np.linspace(50, 110, 500)
adj_range = raw_to_adjusted(raw_range)
ax.plot(raw_range, adj_range, 'b-', linewidth=2, label='Adjustment function')
ax.plot([50, 110], [50, 110], 'k--', alpha=0.3, label='Identity line')

# Annotate median and mean
med_raw = df['raw_score'].median()
mean_raw = df['raw_score'].mean()
med_adj = raw_to_adjusted(med_raw)
mean_adj = raw_to_adjusted(mean_raw)

ax.plot(med_raw, med_adj, 'ro', ms=10, zorder=5)
ax.annotate(f'Median\n({med_raw:.1f}, {med_adj:.1f})',
            xy=(med_raw, med_adj), xytext=(med_raw-8, med_adj+3),
            fontsize=10, ha='center',
            arrowprops=dict(arrowstyle='->', color='red'))

ax.plot(mean_raw, mean_adj, 'gs', ms=10, zorder=5)
ax.annotate(f'Mean\n({mean_raw:.1f}, {mean_adj:.1f})',
            xy=(mean_raw, mean_adj), xytext=(mean_raw+8, mean_adj-3),
            fontsize=10, ha='center',
            arrowprops=dict(arrowstyle='->', color='green'))

ax.set_xlabel('Raw Score (mean cosine distance x 100)', fontsize=12)
ax.set_ylabel('Adjusted Score', fontsize=12)
ax.set_title('DAT-RU Adjustment Function: raw -> adjusted', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(50, 110)
ax.set_ylim(45, 100)
fig.tight_layout()
fig.savefig(OUT / 'adjustment_function.png', dpi=150)
plt.close()
print("Saved: adjustment_function.png")

# Plot 2: Time-on-task
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.bar(range(len(time_stats)), time_stats['mean_adj'], color='steelblue', alpha=0.8)
ax.set_xlabel('Time-spent decile', fontsize=11)
ax.set_ylabel('Mean Adjusted Score', fontsize=11)
ax.set_title(f'Time-on-task vs Adjusted (r={r_adj_time:.4f})', fontsize=12)
ax.set_xticks(range(len(time_stats)))
ax.set_xticklabels([f'{t:.0f}s' for t in time_stats['mean_time']], rotation=45, fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

ax = axes[1]
ax.bar(range(len(time_stats)), time_stats['mean_raw'], color='coral', alpha=0.8)
ax.set_xlabel('Time-spent decile', fontsize=11)
ax.set_ylabel('Mean Raw Score', fontsize=11)
ax.set_title(f'Time-on-task vs Raw (r={r_raw_time:.4f})', fontsize=12)
ax.set_xticks(range(len(time_stats)))
ax.set_xticklabels([f'{t:.0f}s' for t in time_stats['mean_time']], rotation=45, fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

fig.tight_layout()
fig.savefig(OUT / 'time_on_task.png', dpi=150)
plt.close()
print("Saved: time_on_task.png")

# Plot 3: Embedding verification scatter
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(vdf['raw_from_embeddings'], vdf['raw_from_inverse'], alpha=0.5, s=20)
lims = [min(vdf['raw_from_embeddings'].min(), vdf['raw_from_inverse'].min()) - 1,
        max(vdf['raw_from_embeddings'].max(), vdf['raw_from_inverse'].max()) + 1]
ax.plot(lims, lims, 'r--', alpha=0.5)
ax.set_xlabel('Raw from embeddings', fontsize=12)
ax.set_ylabel('Raw from inverse formula', fontsize=12)
ax.set_title(f'Inverse formula verification (r={corr_raw:.4f}, n={len(vdf)})', fontsize=13)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / 'inverse_verification.png', dpi=150)
plt.close()
print("Saved: inverse_verification.png")

# Plot 4: Raw vs Adjusted distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df['raw_score'], bins=80, color='coral', alpha=0.7, edgecolor='black', linewidth=0.3)
axes[0].set_xlabel('Raw Score', fontsize=11)
axes[0].set_ylabel('Count', fontsize=11)
axes[0].set_title('Distribution of Raw Scores', fontsize=12)
axes[0].axvline(df['raw_score'].median(), color='red', ls='--', label=f"median={df['raw_score'].median():.1f}")
axes[0].axvline(df['raw_score'].mean(), color='green', ls='--', label=f"mean={df['raw_score'].mean():.1f}")
axes[0].legend()

axes[1].hist(df['score'], bins=80, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.3)
axes[1].set_xlabel('Adjusted Score', fontsize=11)
axes[1].set_ylabel('Count', fontsize=11)
axes[1].set_title('Distribution of Adjusted Scores', fontsize=12)
axes[1].axvline(df['score'].median(), color='red', ls='--', label=f"median={df['score'].median():.1f}")
axes[1].axvline(df['score'].mean(), color='green', ls='--', label=f"mean={df['score'].mean():.1f}")
axes[1].legend()

fig.tight_layout()
fig.savefig(OUT / 'score_distributions.png', dpi=150)
plt.close()
print("Saved: score_distributions.png")

# ── 4. Summary table ──────────────────────────────────────────────────────

print("\n" + "="*70)
print("SUMMARY TABLE: Adjusted vs Raw score correlations")
print("="*70)

summary = []
summary.append({
    'Analysis': 'Test-retest (attempt1 vs attempt2)',
    'r_adjusted': r_adj_retest,
    'r_raw': r_raw_retest,
    'n': len(pairs_adj),
    'verdict': ''
})
summary.append({
    'Analysis': 'Word frequency (log freq vs score)',
    'r_adjusted': r_adj_freq,
    'r_raw': r_raw_freq,
    'n': len(df),
    'verdict': ''
})
summary.append({
    'Analysis': 'Time-on-task (time vs score)',
    'r_adjusted': r_adj_time,
    'r_raw': r_raw_time,
    'n': len(df_time),
    'verdict': ''
})

sdf = pd.DataFrame(summary)

# Determine verdict
for i, row in sdf.iterrows():
    r_a = row['r_adjusted']
    r_r = row['r_raw']
    if np.isnan(r_a) or np.isnan(r_r):
        sdf.loc[i, 'verdict'] = 'Insufficient data'
    elif np.sign(r_a) == np.sign(r_r) and abs(r_a - r_r) < 0.05:
        sdf.loc[i, 'verdict'] = 'Preserves (same sign, |delta_r| < 0.05)'
    elif np.sign(r_a) == np.sign(r_r):
        sdf.loc[i, 'verdict'] = f'Preserves direction, magnitude differs (delta_r = {abs(r_a-r_r):.4f})'
    else:
        sdf.loc[i, 'verdict'] = 'ARTIFACT: sign reversal'

print(sdf.to_string(index=False, float_format='%.4f'))

# Additional info
print(f"\nInverse formula verification: r = {corr_raw:.6f}, mean diff = {mean_diff:.4f} +/- {std_diff:.4f}")
print(f"  -> {'VALID' if corr_raw > 0.99 else 'MISMATCH'}: inverse formula recovers raw scores")

# Check nonlinearity effect on variance
print(f"\nRaw score:     mean={df['raw_score'].mean():.2f}, SD={df['raw_score'].std():.2f}, skew={df['raw_score'].skew():.3f}")
print(f"Adjusted score: mean={df['score'].mean():.2f}, SD={df['score'].std():.2f}, skew={df['score'].skew():.3f}")

# Save summary
sdf.to_csv(OUT / 'summary_table.csv', index=False)
print(f"\nSaved summary to {OUT / 'summary_table.csv'}")
print("Done.")
