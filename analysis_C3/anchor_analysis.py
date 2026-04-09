"""
C3: Anchor word effect — does the first word determine the set's quality?
"""
import numpy as np
import pandas as pd
import json
import struct
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from wordfreq import word_frequency
from collections import Counter

OUT = "/Users/teo/Downloads/datcreativity/analysis_C3"
DATA = "/Users/teo/Downloads/datcreativity/data"
CSV = "/Users/teo/Downloads/DAT-RU Results - Sheet1 (1).csv"

# ── Load embeddings ──
print("Loading embeddings...")
with open(os.path.join(DATA, "words.json"), "r") as f:
    word_list = json.load(f)
word2idx = {w: i for i, w in enumerate(word_list)}

with open(os.path.join(DATA, "matrix.bin"), "rb") as f:
    num_words, dim = struct.unpack("II", f.read(8))
    raw = np.frombuffer(f.read(), dtype=np.int8).reshape(num_words, dim).astype(np.float32)

# Normalize rows for cosine distance
norms = np.linalg.norm(raw, axis=1, keepdims=True)
norms[norms == 0] = 1
emb = raw / norms

print(f"Embeddings: {num_words} words x {dim} dims")

# ── Load CSV ──
print("Loading CSV...")
df = pd.read_csv(CSV)
print(f"Rows: {len(df)}")

# Parse words
def parse_words(s):
    return [w.strip().lower() for w in s.split(",")]

df["word_list"] = df["words"].apply(parse_words)
df = df[df["word_list"].apply(len) == 7].copy()
print(f"Rows with 7 words: {len(df)}")

# ── Look up vectors, compute pairwise distances ──
print("Computing vectors and distances...")

def get_vectors(words):
    vecs = []
    for w in words:
        idx = word2idx.get(w)
        if idx is None:
            return None
        vecs.append(emb[idx])
    return np.array(vecs)

def pairwise_cosine_dist(vecs):
    """Returns 7x7 distance matrix (1 - cosine_sim)"""
    sim = vecs @ vecs.T
    return 1.0 - sim

vectors = df["word_list"].apply(get_vectors)
valid_mask = vectors.notna()
print(f"Valid (all 7 words in vocab): {valid_mask.sum()}")

df = df[valid_mask].copy()
vectors = vectors[valid_mask]

# Compute distance matrices
dist_matrices = vectors.apply(pairwise_cosine_dist)

# Compute DAT score = mean of all 21 pairwise distances
def dat_score(dm):
    return dm[np.triu_indices(7, k=1)].mean()

df["dat_score_computed"] = dist_matrices.apply(dat_score)

print(f"\nCorrelation between computed and provided scores: {df['dat_score_computed'].corr(df['score']):.4f}")
print(f"Computed DAT mean: {df['dat_score_computed'].mean():.4f}, Provided score mean: {df['score'].mean():.4f}")

# ═══════════════════════════════════════════
# STEP 2: Positional Analysis
# ═══════════════════════════════════════════
print("\n" + "="*60)
print("STEP 2: Positional Analysis")
print("="*60)

# 2a. Mean distance of each position to other 6
n = len(df)
mean_dist_by_pos = np.zeros((n, 7))
for i, dm in enumerate(dist_matrices):
    for p in range(7):
        mask = [j for j in range(7) if j != p]
        mean_dist_by_pos[i, p] = dm[p, mask].mean()

pos_means = mean_dist_by_pos.mean(axis=0)
pos_sems = mean_dist_by_pos.std(axis=0) / np.sqrt(n)

print("\n2a. Mean distance of each position to other 6:")
for p in range(7):
    print(f"  Position {p+1}: {pos_means[p]:.6f} (SE: {pos_sems[p]:.6f})")

# 2c. Leave-one-out score
loo_scores = np.zeros((n, 7))
for i, dm in enumerate(dist_matrices):
    for p in range(7):
        remaining = [j for j in range(7) if j != p]
        sub = dm[np.ix_(remaining, remaining)]
        loo_scores[i, p] = sub[np.triu_indices(6, k=1)].mean()

loo_means = loo_scores.mean(axis=0)
full_mean = df["dat_score_computed"].mean()
contribution = full_mean - loo_means  # positive = removing hurts score

print("\n2c. Leave-one-out analysis:")
print(f"  Full DAT score: {full_mean:.6f}")
for p in range(7):
    print(f"  Position {p+1}: LOO score = {loo_means[p]:.6f}, contribution = {contribution[p]:.6f}")

# 2d. Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.bar(range(1, 8), pos_means, yerr=pos_sems*1.96, capsize=4, color='steelblue', alpha=0.8)
ax.set_xlabel("Word Position (field order)")
ax.set_ylabel("Mean Distance to Other 6")
ax.set_title("2a. Mean Distance by Position")
ax.set_xticks(range(1, 8))

ax = axes[1]
colors = ['#e74c3c' if p == 0 else '#3498db' for p in range(7)]
ax.bar(range(1, 8), contribution, color=colors, alpha=0.8)
ax.set_xlabel("Word Position (field order)")
ax.set_ylabel("Contribution (full − leave-one-out)")
ax.set_title("2c. Contribution to DAT Score by Position")
ax.set_xticks(range(1, 8))
ax.axhline(0, color='black', linewidth=0.5)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "C3_positional_analysis.png"), dpi=150)
plt.close()
print("Saved C3_positional_analysis.png")

# ═══════════════════════════════════════════
# STEP 3: First Word as Predictor
# ═══════════════════════════════════════════
print("\n" + "="*60)
print("STEP 3: First Word as Predictor")
print("="*60)

# 3a. First word isolation
first_word_isolation = mean_dist_by_pos[:, 0]
df["first_word_isolation"] = first_word_isolation

r, p_val = stats.pearsonr(df["first_word_isolation"], df["dat_score_computed"])
print(f"\n3a. First word isolation vs DAT score:")
print(f"  Pearson r = {r:.4f}, p = {p_val:.2e}")

# Compare with other positions
print("\n  Correlation of each position's isolation with DAT score:")
for p in range(7):
    r_p, _ = stats.pearsonr(mean_dist_by_pos[:, p], df["dat_score_computed"].values)
    print(f"  Position {p+1}: r = {r_p:.4f}")

# 3b. Common vs uncommon first words
first_words = df["word_list"].apply(lambda x: x[0])
fw_counts = Counter(first_words)
top20 = [w for w, _ in fw_counts.most_common(20)]
print(f"\n3b. Top 20 most frequent first words: {top20}")

is_common = first_words.isin(top20)
common_score = df.loc[is_common, "dat_score_computed"]
uncommon_score = df.loc[~is_common, "dat_score_computed"]

t_stat, t_p = stats.ttest_ind(uncommon_score, common_score)
d = (uncommon_score.mean() - common_score.mean()) / np.sqrt((uncommon_score.var() + common_score.var()) / 2)
print(f"  Common first word (n={is_common.sum()}): mean DAT = {common_score.mean():.4f}")
print(f"  Uncommon first word (n=(~is_common).sum()): mean DAT = {uncommon_score.mean():.4f}")
print(f"  t = {t_stat:.3f}, p = {t_p:.2e}, Cohen's d = {d:.4f}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.scatter(df["first_word_isolation"], df["dat_score_computed"], alpha=0.05, s=3, color='steelblue')
ax.set_xlabel("First Word Isolation (mean dist to other 6)")
ax.set_ylabel("DAT Score")
ax.set_title(f"3a. First Word Isolation vs DAT Score (r={r:.3f})")
# Add regression line
m, b = np.polyfit(df["first_word_isolation"], df["dat_score_computed"], 1)
x_line = np.linspace(df["first_word_isolation"].min(), df["first_word_isolation"].max(), 100)
ax.plot(x_line, m*x_line + b, 'r-', linewidth=2)

ax = axes[1]
ax.boxplot([common_score, uncommon_score], labels=["Common\n1st word", "Uncommon\n1st word"])
ax.set_ylabel("DAT Score")
ax.set_title(f"3b. Common vs Uncommon First Word\n(d={d:.3f})")

plt.tight_layout()
plt.savefig(os.path.join(OUT, "C3_first_word_predictor.png"), dpi=150)
plt.close()
print("Saved C3_first_word_predictor.png")

# ═══════════════════════════════════════════
# STEP 4: Word Properties by Position
# ═══════════════════════════════════════════
print("\n" + "="*60)
print("STEP 4: Word Properties by Position")
print("="*60)

# 4a. Word length by position
word_lengths = np.array([[len(w) for w in wl] for wl in df["word_list"]])
len_means = word_lengths.mean(axis=0)
len_sems = word_lengths.std(axis=0) / np.sqrt(n)

print("\n4a. Word length by position:")
for p in range(7):
    print(f"  Position {p+1}: {len_means[p]:.3f} chars (SE: {len_sems[p]:.4f})")

# 4b. Corpus frequency by position
print("\n4b. Computing corpus frequencies...")
word_freqs = np.array([[word_frequency(w, 'ru') for w in wl] for wl in df["word_list"]])
freq_means = word_freqs.mean(axis=0)
freq_sems = word_freqs.std(axis=0) / np.sqrt(n)

# Use log frequency for display
log_freqs = np.log10(word_freqs + 1e-9)
log_freq_means = log_freqs.mean(axis=0)
log_freq_sems = log_freqs.std(axis=0) / np.sqrt(n)

print("  Log10 frequency by position:")
for p in range(7):
    print(f"  Position {p+1}: {log_freq_means[p]:.4f} (SE: {log_freq_sems[p]:.5f})")

# 4c. Distance to centroid by position
print("\n4c. Distance to centroid by position:")
centroid_dists = np.zeros((n, 7))
for i, vecs_row in enumerate(vectors):
    centroid = vecs_row.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-10)
    for p in range(7):
        centroid_dists[i, p] = 1.0 - np.dot(vecs_row[p], centroid)

cd_means = centroid_dists.mean(axis=0)
cd_sems = centroid_dists.std(axis=0) / np.sqrt(n)

for p in range(7):
    print(f"  Position {p+1}: {cd_means[p]:.6f} (SE: {cd_sems[p]:.6f})")

# Plot all three
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
ax.bar(range(1, 8), len_means, yerr=len_sems*1.96, capsize=4, color='#2ecc71', alpha=0.8)
ax.set_xlabel("Word Position")
ax.set_ylabel("Mean Word Length (chars)")
ax.set_title("4a. Word Length by Position")
ax.set_xticks(range(1, 8))

ax = axes[1]
ax.bar(range(1, 8), log_freq_means, yerr=log_freq_sems*1.96, capsize=4, color='#e67e22', alpha=0.8)
ax.set_xlabel("Word Position")
ax.set_ylabel("Mean Log10 Frequency")
ax.set_title("4b. Corpus Frequency by Position")
ax.set_xticks(range(1, 8))

ax = axes[2]
colors_cd = ['#e74c3c' if p == 0 else '#9b59b6' for p in range(7)]
ax.bar(range(1, 8), cd_means, yerr=cd_sems*1.96, capsize=4, color=colors_cd, alpha=0.8)
ax.set_xlabel("Word Position")
ax.set_ylabel("Cosine Distance to Centroid")
ax.set_title("4c. Distance to Semantic Centroid")
ax.set_xticks(range(1, 8))

plt.tight_layout()
plt.savefig(os.path.join(OUT, "C3_word_properties.png"), dpi=150)
plt.close()
print("Saved C3_word_properties.png")

# ═══════════════════════════════════════════
# STEP 5: Sequential Divergence
# ═══════════════════════════════════════════
print("\n" + "="*60)
print("STEP 5: Sequential Divergence")
print("="*60)

consec_dists = np.zeros((n, 6))
for i, dm in enumerate(dist_matrices):
    for t in range(6):
        consec_dists[i, t] = dm[t, t+1]

consec_means = consec_dists.mean(axis=0)
consec_sems = consec_dists.std(axis=0) / np.sqrt(n)

print("\n5a-b. Mean consecutive pairwise distance:")
for t in range(6):
    print(f"  Transition {t+1}→{t+2}: {consec_means[t]:.6f} (SE: {consec_sems[t]:.6f})")

# Trend test
slope, intercept, r_val, p_val, se = stats.linregress(range(6), consec_means)
print(f"\n  Linear trend: slope = {slope:.6f}, r = {r_val:.4f}, p = {p_val:.4f}")

fig, ax = plt.subplots(figsize=(8, 5))
labels = [f"{t+1}→{t+2}" for t in range(6)]
ax.bar(range(6), consec_means, yerr=consec_sems*1.96, capsize=4, color='#1abc9c', alpha=0.8)
ax.set_xticks(range(6))
ax.set_xticklabels(labels)
ax.set_xlabel("Transition")
ax.set_ylabel("Mean Cosine Distance")
ax.set_title(f"5. Sequential Divergence\n(slope={slope:.5f}, r={r_val:.3f}, p={p_val:.3f})")
ax.plot(range(6), intercept + slope * np.arange(6), 'r--', linewidth=2)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "C3_sequential_divergence.png"), dpi=150)
plt.close()
print("Saved C3_sequential_divergence.png")

# ═══════════════════════════════════════════
# STEP 6: Risky First Word Analysis
# ═══════════════════════════════════════════
print("\n" + "="*60)
print("STEP 6: Risky First Word Analysis")
print("="*60)

# 6a. Define risky = bottom 25% corpus frequency for first word
first_word_freq = word_freqs[:, 0]
freq_25 = np.percentile(first_word_freq, 25)
risky = first_word_freq <= freq_25

print(f"\n6a. Risky threshold (25th percentile freq): {freq_25:.2e}")
print(f"  Risky first words: {risky.sum()} ({100*risky.mean():.1f}%)")
print(f"  Non-risky: {(~risky).sum()}")

# 6b. Score comparison
risky_scores = df["dat_score_computed"].values[risky]
safe_scores = df["dat_score_computed"].values[~risky]
t_stat, t_p = stats.ttest_ind(risky_scores, safe_scores)
d_risky = (risky_scores.mean() - safe_scores.mean()) / np.sqrt((risky_scores.var() + safe_scores.var()) / 2)

print(f"\n6b. Risky vs safe first word:")
print(f"  Risky mean DAT: {risky_scores.mean():.4f}")
print(f"  Safe mean DAT:  {safe_scores.mean():.4f}")
print(f"  t = {t_stat:.3f}, p = {t_p:.2e}, d = {d_risky:.4f}")

# 6c. Interaction: risky × time_spent
time_spent = df["time_spent"].values
median_time = np.median(time_spent)

risky_long = risky & (time_spent > median_time)
risky_short = risky & (time_spent <= median_time)
safe_long = (~risky) & (time_spent > median_time)
safe_short = (~risky) & (time_spent <= median_time)

groups = {
    "Safe + Short":  df["dat_score_computed"].values[safe_short],
    "Safe + Long":   df["dat_score_computed"].values[safe_long],
    "Risky + Short": df["dat_score_computed"].values[risky_short],
    "Risky + Long":  df["dat_score_computed"].values[risky_long],
}

print(f"\n6c. Interaction (median time = {median_time:.0f}s):")
for label, scores in groups.items():
    print(f"  {label:20s}: n={len(scores):5d}, mean={scores.mean():.4f}")

# 2-way ANOVA-like: compute interaction
from scipy.stats import f_oneway
F, p_anova = f_oneway(*groups.values())
print(f"  One-way ANOVA across 4 groups: F={F:.3f}, p={p_anova:.2e}")

# Plot interaction
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.boxplot([risky_scores, safe_scores], labels=["Risky\n1st word", "Safe\n1st word"])
ax.set_ylabel("DAT Score")
ax.set_title(f"6b. Risky vs Safe First Word\n(d={d_risky:.3f}, p={t_p:.2e})")

ax = axes[1]
group_labels = list(groups.keys())
group_means = [v.mean() for v in groups.values()]
group_sems = [v.std()/np.sqrt(len(v)) for v in groups.values()]
colors_int = ['#3498db', '#2980b9', '#e74c3c', '#c0392b']
ax.bar(range(4), group_means, yerr=[s*1.96 for s in group_sems], capsize=4, color=colors_int, alpha=0.8)
ax.set_xticks(range(4))
ax.set_xticklabels(group_labels, rotation=15, ha='right')
ax.set_ylabel("Mean DAT Score")
ax.set_title("6c. Risky First Word × Time Spent Interaction")

plt.tight_layout()
plt.savefig(os.path.join(OUT, "C3_risky_first_word.png"), dpi=150)
plt.close()
print("Saved C3_risky_first_word.png")

# ═══════════════════════════════════════════
# Summary Statistics
# ═══════════════════════════════════════════
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"N valid submissions: {n}")
print(f"\nPosition 1 vs positions 2-7:")
print(f"  Mean distance to others: pos1={pos_means[0]:.5f}, others={pos_means[1:].mean():.5f}")
print(f"  LOO contribution: pos1={contribution[0]:.6f}, others_mean={contribution[1:].mean():.6f}")
print(f"  Word length: pos1={len_means[0]:.3f}, others={len_means[1:].mean():.3f}")
print(f"  Log freq: pos1={log_freq_means[0]:.4f}, others={log_freq_means[1:].mean():.4f}")
print(f"  Centroid dist: pos1={cd_means[0]:.5f}, others={cd_means[1:].mean():.5f}")
print(f"\nFirst word isolation → DAT: r={stats.pearsonr(first_word_isolation, df['dat_score_computed'].values)[0]:.4f}")
print(f"Sequential divergence slope: {slope:.6f}")
print(f"Risky first word effect: d={d_risky:.4f}")

print("\nDone!")
