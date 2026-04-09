#!/usr/bin/env python3
"""C4: Internal semantic structure of DAT word sets."""

import json
import struct
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cosine as cosine_dist
from itertools import combinations
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

OUT = '/Users/teo/Downloads/datcreativity/analysis_C4'

# ── 1. Load data ──────────────────────────────────────────────────────────
print("Loading embeddings...")
with open('/Users/teo/Downloads/datcreativity/data/words.json', 'r') as f:
    words_list = json.load(f)
word2idx = {w: i for i, w in enumerate(words_list)}
N_words = len(words_list)

with open('/Users/teo/Downloads/datcreativity/data/matrix.bin', 'rb') as f:
    n, dim = struct.unpack('II', f.read(8))
    raw = np.frombuffer(f.read(), dtype=np.int8).reshape(n, dim).astype(np.float32)

# Normalize rows for cosine
norms = np.linalg.norm(raw, axis=1, keepdims=True)
norms[norms == 0] = 1
emb = raw / norms
print(f"Embeddings: {n} words x {dim} dims")

print("Loading CSV...")
df = pd.read_csv('/Users/teo/Downloads/DAT-RU Results - Sheet1 (1).csv')
print(f"CSV: {len(df)} rows")

# ── 2. Compute pairwise distances & structural metrics ────────────────────
print("Computing pairwise distances for all submissions...")

pair_indices = list(combinations(range(7), 2))  # 21 pairs

records = []
all_bottleneck_pairs = []
all_dists_by_row = []

for idx, row in df.iterrows():
    if idx % 5000 == 0:
        print(f"  Processing row {idx}...")

    word_str = row['words']
    score = row['score']

    # Parse words
    words_raw = [w.strip().lower() for w in str(word_str).split(',')]
    if len(words_raw) != 7:
        continue

    # Get embeddings
    indices = []
    valid = True
    for w in words_raw:
        if w in word2idx:
            indices.append(word2idx[w])
        else:
            valid = False
            break
    if not valid or len(indices) != 7:
        continue

    vecs = emb[indices]  # 7 x 300

    # Cosine distances: 1 - cosine_similarity
    # Since vectors are normalized: cos_sim = dot product
    sim_matrix = vecs @ vecs.T

    dists = []
    for i, j in pair_indices:
        d = 1.0 - sim_matrix[i, j]
        dists.append(d)

    dists = np.array(dists)

    mean_d = np.mean(dists)
    sd_d = np.std(dists, ddof=1) if len(dists) > 1 else 0
    cv_d = sd_d / mean_d if mean_d > 0 else 0
    min_d = np.min(dists)
    max_d = np.max(dists)
    range_d = max_d - min_d
    skew_d = stats.skew(dists)
    kurt_d = stats.kurtosis(dists)

    # Bottleneck: closest pair
    min_idx = np.argmin(dists)
    bp_i, bp_j = pair_indices[min_idx]
    bottleneck_w1 = words_raw[bp_i]
    bottleneck_w2 = words_raw[bp_j]

    # Potential score: remove closest pair
    remaining = np.delete(dists, min_idx)
    potential_mean = np.mean(remaining)
    improvement = potential_mean - mean_d

    records.append({
        'idx': idx,
        'score': score,
        'mean_dist': mean_d,
        'sd_dist': sd_d,
        'cv_dist': cv_d,
        'min_dist': min_d,
        'max_dist': max_d,
        'range_dist': range_d,
        'skewness': skew_d,
        'kurtosis': kurt_d,
        'bottleneck_w1': bottleneck_w1,
        'bottleneck_w2': bottleneck_w2,
        'bottleneck_dist': min_d,
        'potential_mean': potential_mean,
        'improvement': improvement,
    })
    all_dists_by_row.append(dists)

    pair_key = tuple(sorted([bottleneck_w1, bottleneck_w2]))
    all_bottleneck_pairs.append(pair_key)

res = pd.DataFrame(records)
print(f"Valid submissions: {len(res)}")

# ── 3. Correlations with score ────────────────────────────────────────────
print("\n=== CORRELATIONS WITH SCORE ===")
metrics = ['mean_dist', 'sd_dist', 'cv_dist', 'min_dist', 'max_dist', 'range_dist', 'skewness', 'kurtosis']
corr_results = {}
for m in metrics:
    r, p = stats.pearsonr(res[m], res['score'])
    corr_results[m] = (r, p)
    print(f"  {m:15s}: r = {r:+.4f}, p = {p:.2e}")

# Partial correlation: min_dist vs score, controlling for mean_dist
from numpy.linalg import lstsq

def partial_corr(x, y, z):
    """Partial correlation of x and y, controlling for z."""
    # Residualize x on z
    A = np.column_stack([z, np.ones(len(z))])
    bx = lstsq(A, x, rcond=None)[0]
    rx = x - A @ bx
    by = lstsq(A, y, rcond=None)[0]
    ry = y - A @ by
    return stats.pearsonr(rx, ry)

r_partial, p_partial = partial_corr(
    res['min_dist'].values, res['score'].values, res['mean_dist'].values
)
print(f"\n  Partial corr (min_dist ~ score | mean_dist): r = {r_partial:+.4f}, p = {p_partial:.2e}")

r_partial_sd, p_partial_sd = partial_corr(
    res['sd_dist'].values, res['score'].values, res['mean_dist'].values
)
print(f"  Partial corr (sd_dist ~ score | mean_dist):  r = {r_partial_sd:+.4f}, p = {p_partial_sd:.2e}")

r_partial_cv, p_partial_cv = partial_corr(
    res['cv_dist'].values, res['score'].values, res['mean_dist'].values
)
print(f"  Partial corr (cv_dist ~ score | mean_dist):  r = {r_partial_cv:+.4f}, p = {p_partial_cv:.2e}")

# ── 4. Bottleneck analysis ────────────────────────────────────────────────
print("\n=== BOTTLENECK ANALYSIS ===")
print(f"  Mean improvement if closest pair removed: {res['improvement'].mean():.4f}")
print(f"  Median improvement: {res['improvement'].median():.4f}")
print(f"  Mean bottleneck distance: {res['bottleneck_dist'].mean():.4f}")
print(f"  Mean overall distance:    {res['mean_dist'].mean():.4f}")

# Most common bottleneck pairs
pair_counts = Counter(all_bottleneck_pairs)
print("\n  Top 20 most common bottleneck pairs:")
for pair, count in pair_counts.most_common(20):
    print(f"    {pair[0]:20s} - {pair[1]:20s}: {count:5d} times")

# ── 5. Topology classification ────────────────────────────────────────────
print("\n=== TOPOLOGY CLASSIFICATION ===")

# Simple heuristic-based classification
def classify_topology(cv, skew, kurt):
    if cv < 0.15:
        return 'Uniform'
    elif skew < -0.5:
        return 'Star'  # negative skew = most distances are large, one cluster pulls down
    elif kurt > 0.5 and cv > 0.2:
        return 'Bimodal'
    else:
        return 'Random'

# Also try k-means
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

features = res[['cv_dist', 'skewness', 'kurtosis']].values
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

km = KMeans(n_clusters=4, random_state=42, n_init=10)
res['km_cluster'] = km.fit_predict(features_scaled)

# Heuristic classification
res['topology'] = res.apply(lambda r: classify_topology(r['cv_dist'], r['skewness'], r['kurtosis']), axis=1)

print("Heuristic classification counts:")
print(res['topology'].value_counts())

print("\nMean score by topology (heuristic):")
topo_stats = res.groupby('topology')['score'].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False)
print(topo_stats)

print("\nMean metrics by topology:")
for topo in ['Uniform', 'Star', 'Bimodal', 'Random']:
    sub = res[res['topology'] == topo]
    if len(sub) > 0:
        print(f"\n  {topo} (n={len(sub)}):")
        print(f"    mean_dist: {sub['mean_dist'].mean():.4f}")
        print(f"    cv_dist:   {sub['cv_dist'].mean():.4f}")
        print(f"    skewness:  {sub['skewness'].mean():.4f}")
        print(f"    min_dist:  {sub['min_dist'].mean():.4f}")

# K-means cluster stats
print("\n\nK-means cluster stats:")
km_stats = res.groupby('km_cluster').agg({
    'score': ['mean', 'std', 'count'],
    'cv_dist': 'mean',
    'skewness': 'mean',
    'kurtosis': 'mean',
    'mean_dist': 'mean',
})
print(km_stats)

# ── 6. Optimal structure (top/bottom 5%) ──────────────────────────────────
print("\n=== OPTIMAL STRUCTURE (Top/Bottom 5%) ===")
q95 = res['score'].quantile(0.95)
q05 = res['score'].quantile(0.05)
top5 = res[res['score'] >= q95]
bot5 = res[res['score'] <= q05]

print(f"Top 5% threshold: {q95:.2f}, n={len(top5)}")
print(f"Bottom 5% threshold: {q05:.2f}, n={len(bot5)}")

print("\nTop 5% topology distribution:")
print(top5['topology'].value_counts(normalize=True).round(3))
print("\nBottom 5% topology distribution:")
print(bot5['topology'].value_counts(normalize=True).round(3))

print("\nTop 5% structural metrics:")
for m in metrics:
    print(f"  {m:15s}: {top5[m].mean():.4f} +/- {top5[m].std():.4f}")

print("\nBottom 5% structural metrics:")
for m in metrics:
    print(f"  {m:15s}: {bot5[m].mean():.4f} +/- {bot5[m].std():.4f}")

# Effect sizes (Cohen's d)
print("\nEffect sizes (Cohen's d, top5 vs bottom5):")
for m in metrics:
    pooled_std = np.sqrt((top5[m].std()**2 + bot5[m].std()**2) / 2)
    d = (top5[m].mean() - bot5[m].mean()) / pooled_std if pooled_std > 0 else 0
    print(f"  {m:15s}: d = {d:+.3f}")

# ── 7. Plots ──────────────────────────────────────────────────────────────
print("\nGenerating plots...")
plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})

# 7a. Scatter: min_dist vs score with LOWESS
from statsmodels.nonparametric.smoothers_lowess import lowess

fig, ax = plt.subplots(figsize=(8, 6))
# Subsample for plotting
np.random.seed(42)
plot_idx = np.random.choice(len(res), min(5000, len(res)), replace=False)
plot_sub = res.iloc[plot_idx]

ax.scatter(plot_sub['min_dist'], plot_sub['score'], alpha=0.15, s=8, c='steelblue')
lw = lowess(res['score'].values, res['min_dist'].values, frac=0.1)
ax.plot(lw[:, 0], lw[:, 1], 'r-', linewidth=2.5, label='LOWESS')
r_min = corr_results['min_dist']
ax.set_xlabel('Minimum Pairwise Distance (closest pair)')
ax.set_ylabel('DAT Score')
ax.set_title(f'Weakest Link: min_dist vs Score (r={r_min[0]:.3f})')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUT}/min_dist_vs_score.png')
plt.close()

# 7b. Scatter: cv_dist vs score
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(plot_sub['cv_dist'], plot_sub['score'], alpha=0.15, s=8, c='steelblue')
lw2 = lowess(res['score'].values, res['cv_dist'].values, frac=0.1)
ax.plot(lw2[:, 0], lw2[:, 1], 'r-', linewidth=2.5, label='LOWESS')
r_cv = corr_results['cv_dist']
ax.set_xlabel('Coefficient of Variation of Pairwise Distances')
ax.set_ylabel('DAT Score')
ax.set_title(f'Uniformity: CV vs Score (r={r_cv[0]:.3f})')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUT}/cv_dist_vs_score.png')
plt.close()

# 7c. Violin plots: score by topology
fig, ax = plt.subplots(figsize=(9, 6))
order = topo_stats.index.tolist()
sns.violinplot(data=res, x='topology', y='score', order=order, ax=ax,
               inner='quartile', palette='Set2', cut=0)
for i, topo in enumerate(order):
    sub = res[res['topology'] == topo]
    ax.text(i, ax.get_ylim()[1] - 2, f'n={len(sub)}', ha='center', va='top', fontsize=9)
ax.set_xlabel('Topology Type')
ax.set_ylabel('DAT Score')
ax.set_title('DAT Score by Semantic Set Topology')
plt.tight_layout()
plt.savefig(f'{OUT}/score_by_topology.png')
plt.close()

# 7d. Example heatmaps
def plot_heatmap(row_data, dists_array, title, filename):
    words_raw = [w.strip() for w in str(df.loc[row_data['idx'], 'words']).split(',')]
    n_w = len(words_raw)
    mat = np.zeros((n_w, n_w))
    k = 0
    for i, j in pair_indices:
        mat[i, j] = dists_array[k]
        mat[j, i] = dists_array[k]
        k += 1

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(mat, xticklabels=words_raw, yticklabels=words_raw,
                annot=True, fmt='.2f', cmap='YlOrRd', ax=ax,
                vmin=0, vmax=1.5, square=True)
    ax.set_title(f'{title}\nScore={row_data["score"]:.1f}, CV={row_data["cv_dist"]:.3f}, Skew={row_data["skewness"]:.2f}')
    plt.tight_layout()
    plt.savefig(f'{OUT}/{filename}')
    plt.close()

# Find best examples from each type
for topo_name in ['Uniform', 'Star', 'Bimodal']:
    sub = res[res['topology'] == topo_name].nlargest(1, 'score')
    if len(sub) > 0:
        row_data = sub.iloc[0]
        # Find the index in our records to get the dists
        rec_idx = res.index[res['idx'] == row_data['idx']][0]
        dists_arr = all_dists_by_row[rec_idx]
        plot_heatmap(row_data, dists_arr, f'Top "{topo_name}" Set', f'heatmap_{topo_name.lower()}.png')
        print(f"  Heatmap for {topo_name}: score={row_data['score']:.1f}, words={df.loc[row_data['idx'], 'words']}")

# 7e. Additional: correlation matrix of structural metrics
fig, ax = plt.subplots(figsize=(9, 7))
corr_mat = res[['score'] + metrics].corr()
sns.heatmap(corr_mat, annot=True, fmt='.3f', cmap='RdBu_r', center=0, ax=ax, square=True, vmin=-1, vmax=1)
ax.set_title('Correlation Matrix: Score vs Structural Metrics')
plt.tight_layout()
plt.savefig(f'{OUT}/correlation_matrix.png')
plt.close()

# 7f. Top vs Bottom 5% structural comparison
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
compare_metrics = ['sd_dist', 'cv_dist', 'min_dist', 'max_dist', 'skewness', 'kurtosis']
for ax, m in zip(axes.flat, compare_metrics):
    data = [top5[m].values, bot5[m].values]
    ax.boxplot(data, labels=['Top 5%', 'Bottom 5%'])
    ax.set_title(m)
    ax.set_ylabel(m)
fig.suptitle('Structural Metrics: Top 5% vs Bottom 5% Scoring Sets', fontsize=13)
plt.tight_layout()
plt.savefig(f'{OUT}/top_vs_bottom_structure.png')
plt.close()

# Save summary
print("\n=== SAVING SUMMARY ===")
res.to_csv(f'{OUT}/structural_metrics.csv', index=False)

# Summary stats
summary = {
    'n_valid': len(res),
    'correlations': {m: {'r': corr_results[m][0], 'p': corr_results[m][1]} for m in metrics},
    'partial_corr_min_dist_given_mean': {'r': r_partial, 'p': p_partial},
    'partial_corr_sd_given_mean': {'r': r_partial_sd, 'p': p_partial_sd},
    'partial_corr_cv_given_mean': {'r': r_partial_cv, 'p': p_partial_cv},
    'bottleneck_mean_improvement': float(res['improvement'].mean()),
    'topology_counts': res['topology'].value_counts().to_dict(),
    'topology_mean_scores': res.groupby('topology')['score'].mean().to_dict(),
    'top5_topology_pct': top5['topology'].value_counts(normalize=True).to_dict(),
    'bot5_topology_pct': bot5['topology'].value_counts(normalize=True).to_dict(),
}

with open(f'{OUT}/summary_c4.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print("\nDone! All outputs saved to", OUT)
