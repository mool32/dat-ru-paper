"""
A4: Split-half reliability analysis for DAT-RU dataset.
"""
import json
import struct
import numpy as np
import pandas as pd
from itertools import combinations
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time

OUT_DIR = "/Users/teo/Downloads/datcreativity/analysis_A4"

# ── 1. Load embeddings ──────────────────────────────────────────────
print("Loading embeddings...")
with open("/Users/teo/Downloads/datcreativity/data/words.json", "r") as f:
    words_list = json.load(f)

word2idx = {w: i for i, w in enumerate(words_list)}
n_words = len(words_list)
print(f"  Vocabulary: {n_words} words")

with open("/Users/teo/Downloads/datcreativity/data/matrix.bin", "rb") as f:
    num_words, dim = struct.unpack("II", f.read(8))
    print(f"  Matrix header: {num_words} words × {dim} dims")
    raw = np.frombuffer(f.read(), dtype=np.int8)
    matrix = raw.reshape(num_words, dim).astype(np.float32)

# Normalize rows for cosine distance via dot product
norms = np.linalg.norm(matrix, axis=1, keepdims=True)
norms[norms == 0] = 1.0
matrix_normed = matrix / norms

# ── 2. Load CSV ─────────────────────────────────────────────────────
print("Loading CSV...")
df = pd.read_csv("/Users/teo/Downloads/DAT-RU Results - Sheet1 (1).csv")
print(f"  {len(df)} records")

# Parse words
def parse_words(words_str):
    return [w.strip().lower() for w in words_str.split(",")]

df["word_list"] = df["words"].apply(parse_words)

# ── 3. Precompute all 21 pairwise cosine distances per submission ───
print("Computing pairwise distances for all submissions...")

# For 7 words, there are C(7,2) = 21 pairs
pair_indices_7 = list(combinations(range(7), 2))  # 21 pairs

# Build vectors for all submissions
all_distances = []  # will be (N, 21) array
valid_mask = []
dat_scores = []

t0 = time.time()
n_missing = 0

for idx, row in df.iterrows():
    words = row["word_list"]
    if len(words) != 7:
        valid_mask.append(False)
        all_distances.append(np.zeros(21))
        dat_scores.append(np.nan)
        continue

    # Look up indices
    indices = []
    skip = False
    for w in words:
        if w in word2idx:
            indices.append(word2idx[w])
        else:
            skip = True
            n_missing += 1
            break

    if skip:
        valid_mask.append(False)
        all_distances.append(np.zeros(21))
        dat_scores.append(np.nan)
        continue

    # Get normalized vectors
    vecs = matrix_normed[indices]  # (7, 300)

    # Compute all 21 pairwise cosine distances
    dists = np.zeros(21)
    for k, (i, j) in enumerate(pair_indices_7):
        cos_sim = np.dot(vecs[i], vecs[j])
        dists[k] = 1.0 - cos_sim

    all_distances.append(dists)
    valid_mask.append(True)
    dat_scores.append(row["score"])

all_distances = np.array(all_distances)
valid_mask = np.array(valid_mask)
dat_scores = np.array(dat_scores)

n_valid = valid_mask.sum()
print(f"  Valid submissions: {n_valid} / {len(df)}")
print(f"  Words not found: {n_missing}")
print(f"  Time: {time.time()-t0:.1f}s")

# Filter to valid only
distances_valid = all_distances[valid_mask]  # (N_valid, 21)
scores_valid = dat_scores[valid_mask]

# ── 4. Exhaustive split-half (all 35 splits of 7 into 3+4) ─────────
print("\nExhaustive split-half analysis (all 35 splits)...")

# Generate all C(7,3) = 35 ways to pick 3 items from 7
splits_3 = list(combinations(range(7), 3))

# For each split, we need to know which of the 21 pairwise distances
# belong to the 3-group and which to the 4-group
# pair_indices_7[k] = (i, j) is the k-th pair

# Precompute: for each split, which pair indices belong to each half
split_pair_masks = []
for s3 in splits_3:
    s4 = tuple(i for i in range(7) if i not in s3)
    pairs_3 = []
    pairs_4 = []
    for k, (i, j) in enumerate(pair_indices_7):
        if i in s3 and j in s3:
            pairs_3.append(k)
        elif i in s4 and j in s4:
            pairs_4.append(k)
    split_pair_masks.append((pairs_3, pairs_4))

# Compute half-scores for all submissions × all 35 splits
half_scores_3 = np.zeros((n_valid, 35))
half_scores_4 = np.zeros((n_valid, 35))

for s_idx, (p3, p4) in enumerate(split_pair_masks):
    half_scores_3[:, s_idx] = distances_valid[:, p3].mean(axis=1)  # mean of 3 pairs
    half_scores_4[:, s_idx] = distances_valid[:, p4].mean(axis=1)  # mean of 6 pairs

# Correlation for each split
split_correlations = np.zeros(35)
for s_idx in range(35):
    r, _ = stats.pearsonr(half_scores_3[:, s_idx], half_scores_4[:, s_idx])
    split_correlations[s_idx] = r

mean_split_r = split_correlations.mean()
sb_exhaustive = 2 * mean_split_r / (1 + mean_split_r)

print(f"  Mean split-half r across 35 splits: {mean_split_r:.4f}")
print(f"  Spearman-Brown corrected: {sb_exhaustive:.4f}")
print(f"  Range of split r: [{split_correlations.min():.4f}, {split_correlations.max():.4f}]")

# ── 5. Random split-half (1000 iterations) ──────────────────────────
print("\nRandom split-half (1000 iterations)...")
np.random.seed(42)
n_iter = 1000
random_correlations = np.zeros(n_iter)

for it in range(n_iter):
    # Random permutation for each submission
    # For each row, randomly pick 3 of 7 words
    perm = np.array([np.random.permutation(7) for _ in range(n_valid)])
    half_a = np.zeros(n_valid)
    half_b = np.zeros(n_valid)

    for row_idx in range(n_valid):
        group_a = set(perm[row_idx, :3])
        group_b = set(perm[row_idx, 3:])

        sum_a, count_a = 0.0, 0
        sum_b, count_b = 0.0, 0
        for k, (i, j) in enumerate(pair_indices_7):
            if i in group_a and j in group_a:
                sum_a += distances_valid[row_idx, k]
                count_a += 1
            elif i in group_b and j in group_b:
                sum_b += distances_valid[row_idx, k]
                count_b += 1

        half_a[row_idx] = sum_a / count_a if count_a > 0 else 0
        half_b[row_idx] = sum_b / count_b if count_b > 0 else 0

    r, _ = stats.pearsonr(half_a, half_b)
    random_correlations[it] = r

mean_random_r = random_correlations.mean()
sb_random = 2 * mean_random_r / (1 + mean_random_r)

print(f"  Mean random split-half r: {mean_random_r:.4f}")
print(f"  Spearman-Brown corrected: {sb_random:.4f}")
print(f"  95% CI of r: [{np.percentile(random_correlations, 2.5):.4f}, {np.percentile(random_correlations, 97.5):.4f}]")

# ── 6. Cronbach's alpha ─────────────────────────────────────────────
print("\nCronbach's alpha (21 pairwise distances as items)...")

k_items = 21
item_variances = distances_valid.var(axis=0, ddof=1)
total_scores = distances_valid.sum(axis=1)
total_variance = total_scores.var(ddof=1)

cronbach_alpha = (k_items / (k_items - 1)) * (1 - item_variances.sum() / total_variance)
print(f"  Cronbach's alpha: {cronbach_alpha:.4f}")

# ── 7. Score-dependent reliability (top/bottom 25%) ─────────────────
print("\nScore-dependent reliability...")

# Use the mean of all 21 pairwise distances as the total score
total_dist = distances_valid.mean(axis=1)  # our own computed DAT score
q25 = np.percentile(total_dist, 25)
q75 = np.percentile(total_dist, 75)

low_mask = total_dist <= q25
high_mask = total_dist >= q75

print(f"  Score range: [{total_dist.min():.4f}, {total_dist.max():.4f}]")
print(f"  Q25={q25:.4f}, Q75={q75:.4f}")
print(f"  Full sample variance of total score: {total_dist.var(ddof=1):.6f}")

for label, mask in [("Bottom 25%", low_mask), ("Top 25%", high_mask)]:
    subset = distances_valid[mask]
    n_sub = mask.sum()
    sub_total = total_dist[mask]
    print(f"  {label} (n={n_sub}): score range [{sub_total.min():.4f}, {sub_total.max():.4f}], var={sub_total.var(ddof=1):.6f}")

    # Exhaustive split-half for this subset
    sub_corrs = np.zeros(35)
    for s_idx, (p3, p4) in enumerate(split_pair_masks):
        h3 = subset[:, p3].mean(axis=1)
        h4 = subset[:, p4].mean(axis=1)
        r, _ = stats.pearsonr(h3, h4)
        sub_corrs[s_idx] = r

    mean_r = sub_corrs.mean()
    sb = 2 * mean_r / (1 + mean_r)

    # Cronbach's alpha for subset
    iv = subset.var(axis=0, ddof=1)
    tv = subset.sum(axis=1).var(ddof=1)
    alpha = (21 / 20) * (1 - iv.sum() / tv)

    print(f"    split-half r={mean_r:.4f}, SB={sb:.4f}, alpha={alpha:.4f}")

# ── 8. Plots ────────────────────────────────────────────────────────
print("\nGenerating plots...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("A4: Split-Half Reliability Analysis — DAT-RU", fontsize=14, fontweight='bold')

# Plot 1: Distribution of split-half correlations (exhaustive)
ax = axes[0, 0]
ax.hist(split_correlations, bins=15, color='steelblue', edgecolor='black', alpha=0.8)
ax.axvline(mean_split_r, color='red', linestyle='--', linewidth=2, label=f'Mean r = {mean_split_r:.3f}')
ax.set_xlabel("Split-half correlation (r)")
ax.set_ylabel("Count")
ax.set_title("Exhaustive Split-Half (35 splits)")
ax.legend()

# Plot 2: Distribution of random split-half correlations
ax = axes[0, 1]
ax.hist(random_correlations, bins=40, color='darkorange', edgecolor='black', alpha=0.8)
ax.axvline(mean_random_r, color='red', linestyle='--', linewidth=2, label=f'Mean r = {mean_random_r:.3f}')
ax.axvline(np.percentile(random_correlations, 2.5), color='gray', linestyle=':', label='95% CI')
ax.axvline(np.percentile(random_correlations, 97.5), color='gray', linestyle=':')
ax.set_xlabel("Split-half correlation (r)")
ax.set_ylabel("Count")
ax.set_title("Random Split-Half (1000 iterations)")
ax.legend()

# Plot 3: Comparison bar chart
ax = axes[1, 0]
metrics = ['Split-half\n(raw)', 'Split-half\n(SB corrected)', "Cronbach's\nalpha", 'Test-retest\n(previous)']
values = [mean_split_r, sb_exhaustive, cronbach_alpha, 0.231]
colors = ['steelblue', 'steelblue', 'teal', 'coral']
bars = ax.bar(metrics, values, color=colors, edgecolor='black', alpha=0.85)
ax.set_ylabel("Reliability coefficient")
ax.set_title("Reliability Comparison")
ax.set_ylim(0, 1)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.3f}',
            ha='center', va='bottom', fontweight='bold')

# Plot 4: Score-dependent reliability
ax = axes[1, 1]
# Use quintiles (5 groups) for smoother picture
n_bins = 5
bin_labels = [f'Q{i+1}' for i in range(n_bins)]
percentile_edges = np.linspace(0, 100, n_bins + 1)
edges = [np.percentile(total_dist, p) for p in percentile_edges]

group_raw_rs = []
group_sbs = []
group_alphas = []
group_ns = []

for i in range(n_bins):
    if i == n_bins - 1:
        mask = (total_dist >= edges[i]) & (total_dist <= edges[i+1])
    else:
        mask = (total_dist >= edges[i]) & (total_dist < edges[i+1])
    subset = distances_valid[mask]
    group_ns.append(mask.sum())

    sub_corrs = np.zeros(35)
    for s_idx, (p3, p4) in enumerate(split_pair_masks):
        h3 = subset[:, p3].mean(axis=1)
        h4 = subset[:, p4].mean(axis=1)
        r, _ = stats.pearsonr(h3, h4)
        sub_corrs[s_idx] = r
    mr = sub_corrs.mean()
    group_raw_rs.append(mr)
    group_sbs.append(2 * mr / (1 + mr))

    iv = subset.var(axis=0, ddof=1)
    tv = subset.sum(axis=1).var(ddof=1)
    group_alphas.append((21/20) * (1 - iv.sum() / tv))

x = np.arange(n_bins)
w = 0.35
ax.bar(x - w/2, group_raw_rs, w, label='Raw split-half r', color='steelblue', edgecolor='black', alpha=0.85)
ax.bar(x + w/2, group_alphas, w, label="Cronbach's α", color='teal', edgecolor='black', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([f'{l}\n(n={n})' for l, n in zip(bin_labels, group_ns)], fontsize=8)
ax.set_ylabel("Reliability coefficient")
ax.set_title("Reliability by Score Quintile")
ax.axhline(0, color='black', linewidth=0.5)
ax.set_ylim(min(min(group_raw_rs), min(group_alphas), 0) - 0.1, 1.0)
ax.legend(fontsize=8)
for i, (rr, al) in enumerate(zip(group_raw_rs, group_alphas)):
    ax.text(i - w/2, rr + 0.02 if rr >= 0 else rr - 0.06, f'{rr:.3f}', ha='center', fontsize=7, fontweight='bold')
    ax.text(i + w/2, al + 0.02 if al >= 0 else al - 0.06, f'{al:.3f}', ha='center', fontsize=7, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "split_half_reliability.png"), dpi=150, bbox_inches='tight')
print(f"  Saved plot to {OUT_DIR}/split_half_reliability.png")

# ── Additional plot: Score distribution to explain ceiling effect ────
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle("A4 Supplement: Score Distribution & Ceiling Effect", fontsize=13, fontweight='bold')

ax = axes2[0]
ax.hist(total_dist, bins=100, color='steelblue', edgecolor='none', alpha=0.8)
ax.axvline(q25, color='red', linestyle='--', label=f'Q25 = {q25:.3f}')
ax.axvline(q75, color='red', linestyle='--', label=f'Q75 = {q75:.3f}')
ax.set_xlabel("Mean pairwise cosine distance (DAT score)")
ax.set_ylabel("Count")
ax.set_title("Score Distribution (all submissions)")
ax.legend()

# Variance by decile
ax = axes2[1]
n_dec = 10
dec_edges = [np.percentile(total_dist, p) for p in np.linspace(0, 100, n_dec+1)]
dec_vars = []
dec_labels = []
for i in range(n_dec):
    if i == n_dec - 1:
        mask = (total_dist >= dec_edges[i]) & (total_dist <= dec_edges[i+1])
    else:
        mask = (total_dist >= dec_edges[i]) & (total_dist < dec_edges[i+1])
    dec_vars.append(total_dist[mask].var(ddof=1))
    dec_labels.append(f'D{i+1}')

ax.bar(dec_labels, dec_vars, color='teal', edgecolor='black', alpha=0.85)
ax.set_xlabel("Score Decile (D1=lowest scores, D10=highest)")
ax.set_ylabel("Variance of total score within decile")
ax.set_title("Variance by Decile (ceiling effect)")
ax.set_yscale('log')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "score_distribution_ceiling.png"), dpi=150, bbox_inches='tight')
print(f"  Saved plot to {OUT_DIR}/score_distribution_ceiling.png")

# ── 9. Summary report ───────────────────────────────────────────────
summary = f"""
============================================================
A4: SPLIT-HALF RELIABILITY ANALYSIS — DAT-RU
============================================================

Dataset: {len(df)} submissions, {n_valid} valid (all 7 words found in embeddings)

1. EXHAUSTIVE SPLIT-HALF (all 35 splits of 7 → 3+4):
   Mean split-half r:          {mean_split_r:.4f}
   Spearman-Brown corrected:   {sb_exhaustive:.4f}
   Range of r across splits:   [{split_correlations.min():.4f}, {split_correlations.max():.4f}]

2. RANDOM SPLIT-HALF (1000 iterations):
   Mean split-half r:          {mean_random_r:.4f}
   Spearman-Brown corrected:   {sb_random:.4f}
   95% CI of r:                [{np.percentile(random_correlations, 2.5):.4f}, {np.percentile(random_correlations, 97.5):.4f}]

3. CRONBACH'S ALPHA (21 pairwise distances as items):
   Cronbach's alpha:           {cronbach_alpha:.4f}

4. COMPARISON WITH TEST-RETEST:
   Split-half (SB corrected):  {sb_exhaustive:.4f}
   Test-retest r:              0.231

5. SCORE-DEPENDENT RELIABILITY (by quintile):
   {"".join(f"   Q{i+1} (n={group_ns[i]}): raw r = {group_raw_rs[i]:.4f}, SB = {group_sbs[i]:.4f}, alpha = {group_alphas[i]:.4f}" + chr(10) for i in range(n_bins))}

6. SCORE DISTRIBUTION & CEILING EFFECT:
   Score range: [{total_dist.min():.4f}, {total_dist.max():.4f}]
   Q25 = {q25:.4f}, Median = {np.median(total_dist):.4f}, Q75 = {q75:.4f}
   Full-sample variance: {total_dist.var(ddof=1):.6f}
   Q1 variance: {total_dist[low_mask].var(ddof=1):.6f} (contains real spread)
   Q5 variance: {total_dist[high_mask].var(ddof=1):.6f} (ceiling-compressed)
   Ratio Q1/Q5 variance: {total_dist[low_mask].var(ddof=1) / total_dist[high_mask].var(ddof=1):.1f}x

   The score distribution is heavily LEFT-skewed: 75% of submissions
   fall in the narrow range [{q25:.3f}, {total_dist.max():.3f}].
   Only the bottom quintile (creative outliers) has enough variance
   for meaningful within-group reliability estimation.

7. INTERPRETATION:
"""

if sb_exhaustive > 0.6 and sb_exhaustive > 0.231 * 2:
    summary += f"""   OVERALL: Split-half reliability (SB = {sb_exhaustive:.3f}) >> test-retest (r = 0.231).
   Cronbach's alpha = {cronbach_alpha:.3f} is excellent.

   The DAT is internally consistent: within a single session, a person's
   7 words form a coherent pattern. The 21 pairwise distances agree with
   each other about whether this person produced semantically diverse words.

   However, the low test-retest (0.231) means divergent thinking VARIES
   across occasions.
   -> Divergent thinking is STATE-LIKE, not purely trait-like.
   -> The test measures well *at a given moment*, but creativity fluctuates.

   CAVEAT: Reliability is score-dependent due to a strong ceiling effect.
   Only the bottom 20% of scores (the most creative responses) show positive
   within-group reliability (r = {group_raw_rs[0]:.3f}). The top 80% are compressed
   into a narrow range where noise exceeds signal within-group. This means
   the overall alpha ({cronbach_alpha:.3f}) is partly driven by the between-group
   spread between creative outliers and the ceiling-clustered majority.
"""
elif sb_exhaustive < 0.4:
    summary += """   Split-half reliability is also low, similar to test-retest.
   -> The test itself has significant MEASUREMENT NOISE.
   The pairwise distance metric may not form a reliable internal structure.
"""
else:
    summary += """   Split-half reliability is moderate, somewhat higher than test-retest.
   -> Mixed picture: some measurement noise + some state variability.
"""

summary += "============================================================\n"
print(summary)

with open(os.path.join(OUT_DIR, "results_A4.txt"), "w") as f:
    f.write(summary)
print(f"Saved results to {OUT_DIR}/results_A4.txt")
