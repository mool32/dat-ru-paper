"""
C1: Word Frequency Effect Analysis for DAT-RU
"""
import pandas as pd
import numpy as np
import json
import struct
from scipy import stats
from wordfreq import word_frequency, zipf_frequency
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter

OUT = Path('/Users/teo/Downloads/datcreativity/analysis_C1')
DATA = Path('/Users/teo/Downloads/datcreativity/data')

# ─── 1. Load CSV, parse words ───
print("=== Loading data ===")
df = pd.read_csv('/Users/teo/Downloads/DAT-RU Results - Sheet1 (1).csv')
print(f"Total submissions: {len(df)}")

# Parse words
def parse_words(w):
    if pd.isna(w):
        return []
    return [x.strip().lower() for x in str(w).split(',') if x.strip()]

df['word_list'] = df['words'].apply(parse_words)
df = df[df['word_list'].apply(len) == 7].copy()
print(f"Submissions with exactly 7 words: {len(df)}")

# Count word frequencies in DAT
all_words = [w for wl in df['word_list'] for w in wl]
word_counts = Counter(all_words)
unique_words = sorted(word_counts.keys())
print(f"Total word tokens: {len(all_words)}")
print(f"Unique lemmas: {len(unique_words)}")

# ─── 2. Load embeddings ───
print("\n=== Loading embeddings ===")
with open(DATA / 'words.json') as f:
    vocab = json.load(f)
vocab_idx = {w: i for i, w in enumerate(vocab)}

with open(DATA / 'matrix.bin', 'rb') as f:
    nw, dim = struct.unpack('II', f.read(8))
    raw = np.frombuffer(f.read(nw * dim), dtype=np.int8).reshape(nw, dim).astype(np.float32)

# Normalize rows for cosine similarity
norms = np.linalg.norm(raw, axis=1, keepdims=True)
norms[norms == 0] = 1
emb_normed = raw / norms
print(f"Embedding matrix: {nw} x {dim}")

# ─── 3. Build word stats table ───
print("\n=== Building word stats ===")

# For mean score contribution: for each word, average cosine distance
# when it appears with other words in a submission
# We'll compute this efficiently

# First, get embeddings for all DAT words
dat_words_in_vocab = [w for w in unique_words if w in vocab_idx]
print(f"DAT words found in embedding vocab: {len(dat_words_in_vocab)} / {len(unique_words)}")

# Precompute: for each word, collect all submissions it appears in,
# then compute mean cosine distance to co-occurring words
print("Computing mean score contributions...")

# Build word -> list of submission indices
word_to_subs = {}
for i, wl in enumerate(df['word_list']):
    for w in wl:
        if w not in word_to_subs:
            word_to_subs[w] = []
        word_to_subs[w].append(i)

# For each word, compute mean cosine distance to its co-occurring words
word_mean_contrib = {}
word_lists_arr = df['word_list'].values
scores_arr = df['score'].values

# More efficient: for each word, sample up to 500 submissions
np.random.seed(42)
for wi, w in enumerate(dat_words_in_vocab):
    if wi % 2000 == 0:
        print(f"  Processing word {wi}/{len(dat_words_in_vocab)}...")
    if w not in vocab_idx:
        continue
    w_emb = emb_normed[vocab_idx[w]]
    subs = word_to_subs.get(w, [])
    if not subs:
        continue

    # Sample if too many
    if len(subs) > 500:
        subs_sample = np.random.choice(subs, 500, replace=False)
    else:
        subs_sample = subs

    distances = []
    for si in subs_sample:
        wl = word_lists_arr[si]
        for other in wl:
            if other != w and other in vocab_idx:
                other_emb = emb_normed[vocab_idx[other]]
                cos_sim = np.dot(w_emb, other_emb)
                distances.append(1 - cos_sim)

    if distances:
        word_mean_contrib[w] = np.mean(distances)

print(f"Computed contributions for {len(word_mean_contrib)} words")

# ─── 4. Build master dataframe ───
print("\n=== Building master word table ===")
rows = []
for w in unique_words:
    wf = word_frequency(w, 'ru')
    log_wf = np.log10(wf) if wf > 0 else np.nan
    zf = zipf_frequency(w, 'ru')
    rows.append({
        'word': w,
        'dat_freq': word_counts[w],
        'corpus_freq': wf,
        'log_corpus_freq': log_wf,
        'zipf_freq': zf,
        'mean_score_contrib': word_mean_contrib.get(w, np.nan),
        'in_vocab': w in vocab_idx,
    })

wdf = pd.DataFrame(rows)
wdf['dat_freq_rank'] = wdf['dat_freq'].rank(ascending=False, method='min')
wdf['corpus_freq_rank'] = wdf['corpus_freq'].rank(ascending=False, method='min')

# Filter to words that have corpus frequency
wdf_with_corpus = wdf[wdf['corpus_freq'] > 0].copy()
print(f"Words with corpus frequency > 0: {len(wdf_with_corpus)} / {len(wdf)}")

# ─── 5. ANALYSIS ───
print("\n" + "="*60)
print("ANALYSIS RESULTS")
print("="*60)

# 5a. Scatter: corpus_freq vs DAT_freq (log-log)
fig, ax = plt.subplots(figsize=(10, 8))
mask = wdf_with_corpus['dat_freq'] > 0
x = np.log10(wdf_with_corpus.loc[mask, 'corpus_freq'])
y = np.log10(wdf_with_corpus.loc[mask, 'dat_freq'])
ax.scatter(x, y, alpha=0.1, s=5, color='steelblue')
# Fit line
slope, intercept, r, p, se = stats.linregress(x, y)
xline = np.linspace(x.min(), x.max(), 100)
ax.plot(xline, slope * xline + intercept, 'r-', linewidth=2,
        label=f'r={r:.3f}, slope={slope:.2f}')
ax.set_xlabel('log10(Corpus Frequency)', fontsize=12)
ax.set_ylabel('log10(DAT Frequency)', fontsize=12)
ax.set_title('Corpus Frequency vs DAT Usage Frequency', fontsize=14)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(OUT / 'c1a_corpus_vs_dat_freq.png', dpi=150)
plt.close()
print(f"\n[5a] Corpus freq vs DAT freq: r={r:.4f}, slope={slope:.3f}, p={p:.2e}")

# 5b. Scatter: corpus_freq vs mean_score_contribution
wdf_score = wdf_with_corpus.dropna(subset=['mean_score_contrib']).copy()
fig, ax = plt.subplots(figsize=(10, 8))
x2 = wdf_score['log_corpus_freq']
y2 = wdf_score['mean_score_contrib']
ax.scatter(x2, y2, alpha=0.1, s=5, color='coral')
slope2, intercept2, r2, p2, se2 = stats.linregress(x2, y2)
xline2 = np.linspace(x2.min(), x2.max(), 100)
ax.plot(xline2, slope2 * xline2 + intercept2, 'k-', linewidth=2,
        label=f'r={r2:.3f}, slope={slope2:.4f}')
ax.set_xlabel('log10(Corpus Frequency)', fontsize=12)
ax.set_ylabel('Mean Cosine Distance (Score Contribution)', fontsize=12)
ax.set_title('Corpus Frequency vs Word Score Contribution', fontsize=14)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(OUT / 'c1b_corpus_freq_vs_score_contrib.png', dpi=150)
plt.close()
print(f"[5b] Corpus freq vs score contribution: r={r2:.4f}, slope={slope2:.5f}, p={p2:.2e}")

# 5c. Per-submission: average corpus frequency of 7 words vs score
print("\nComputing per-submission average corpus frequency...")
def avg_corpus_freq(wl):
    freqs = [word_frequency(w, 'ru') for w in wl]
    freqs = [f for f in freqs if f > 0]
    if not freqs:
        return np.nan
    return np.mean(np.log10(freqs))

def avg_zipf(wl):
    zfs = [zipf_frequency(w, 'ru') for w in wl]
    return np.mean(zfs)

df['avg_log_corpus_freq'] = df['word_list'].apply(avg_corpus_freq)
df['avg_zipf'] = df['word_list'].apply(avg_zipf)

df_valid = df.dropna(subset=['avg_log_corpus_freq', 'score'])
r3, p3 = stats.pearsonr(df_valid['avg_log_corpus_freq'], df_valid['score'])
r3s, p3s = stats.spearmanr(df_valid['avg_log_corpus_freq'], df_valid['score'])

fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(df_valid['avg_log_corpus_freq'], df_valid['score'], alpha=0.05, s=3, color='purple')
slope3, intercept3, _, _, _ = stats.linregress(df_valid['avg_log_corpus_freq'], df_valid['score'])
xr = np.linspace(df_valid['avg_log_corpus_freq'].min(), df_valid['avg_log_corpus_freq'].max(), 100)
ax.plot(xr, slope3*xr + intercept3, 'r-', linewidth=2,
        label=f'Pearson r={r3:.3f}, Spearman rho={r3s:.3f}')
ax.set_xlabel('Mean log10(Corpus Frequency) of 7 Words', fontsize=12)
ax.set_ylabel('DAT Score', fontsize=12)
ax.set_title('Average Word Rarity vs DAT Score', fontsize=14)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(OUT / 'c1c_avg_rarity_vs_score.png', dpi=150)
plt.close()
print(f"[5c] Avg corpus freq vs score: Pearson r={r3:.4f} (p={p3:.2e}), Spearman rho={r3s:.4f} (p={p3s:.2e})")

# 5d. Zipf analysis
print("\n[5d] Zipf Analysis")
# DAT word frequency: rank vs frequency
dat_freqs = sorted(word_counts.values(), reverse=True)
ranks = np.arange(1, len(dat_freqs) + 1)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# DAT Zipf
ax = axes[0]
ax.scatter(np.log10(ranks), np.log10(dat_freqs), s=3, alpha=0.5, color='steelblue')
# Fit Zipf exponent (log f = -alpha * log r + C)
log_r = np.log10(ranks)
log_f = np.log10(dat_freqs)
# Use only top 5000 words for fit (tail deviates)
n_fit = min(5000, len(log_r))
slope_z, intercept_z, r_z, _, _ = stats.linregress(log_r[:n_fit], log_f[:n_fit])
ax.plot(log_r, slope_z * log_r + intercept_z, 'r-', linewidth=2,
        label=f'Zipf alpha={-slope_z:.3f} (top {n_fit})')
ax.set_xlabel('log10(Rank)', fontsize=12)
ax.set_ylabel('log10(Frequency)', fontsize=12)
ax.set_title('DAT Word Frequency: Zipf Plot', fontsize=14)
ax.legend(fontsize=11)

# Compare with corpus Zipf: get corpus frequencies for DAT words, rank them
corpus_freqs_dat = [(w, word_frequency(w, 'ru')) for w in unique_words]
corpus_freqs_dat = [(w, f) for w, f in corpus_freqs_dat if f > 0]
corpus_freqs_dat.sort(key=lambda x: x[1], reverse=True)
corpus_f = [f for _, f in corpus_freqs_dat]
corpus_ranks = np.arange(1, len(corpus_f) + 1)

ax2 = axes[1]
ax2.scatter(np.log10(corpus_ranks), np.log10(corpus_f), s=3, alpha=0.5, color='coral',
            label='Corpus freq of DAT words')
log_cr = np.log10(corpus_ranks)
log_cf = np.log10(corpus_f)
n_fit2 = min(5000, len(log_cr))
slope_c, intercept_c, r_c, _, _ = stats.linregress(log_cr[:n_fit2], log_cf[:n_fit2])
ax2.plot(log_cr, slope_c * log_cr + intercept_c, 'k-', linewidth=2,
         label=f'Zipf alpha={-slope_c:.3f} (top {n_fit2})')
ax2.set_xlabel('log10(Rank)', fontsize=12)
ax2.set_ylabel('log10(Corpus Frequency)', fontsize=12)
ax2.set_title('Corpus Frequency of DAT Words: Zipf Plot', fontsize=14)
ax2.legend(fontsize=11)

plt.tight_layout()
plt.savefig(OUT / 'c1d_zipf_analysis.png', dpi=150)
plt.close()
print(f"  DAT Zipf exponent: alpha = {-slope_z:.3f} (r={r_z:.4f})")
print(f"  Corpus Zipf exponent (DAT words): alpha = {-slope_c:.3f} (r={r_c:.4f})")

# 5e. Overused/underused analysis
print("\n[5e] Overused/Underused Words")
wdf_ranked = wdf_with_corpus[wdf_with_corpus['dat_freq'] > 0].copy()
# Ratio: corpus_freq_rank / dat_freq_rank
# If ratio > 1: word is used MORE in DAT than expected from corpus (overused)
# If ratio < 1: word is used LESS in DAT than expected (underused)
wdf_ranked['rank_ratio'] = wdf_ranked['corpus_freq_rank'] / wdf_ranked['dat_freq_rank']
# Also: log ratio
wdf_ranked['log_rank_ratio'] = np.log10(wdf_ranked['rank_ratio'])

# Overused: high rank_ratio (corpus rank is high = rare in corpus, but DAT rank is low = common in DAT)
# Wait, let's think again:
# dat_freq_rank=1 means most frequent in DAT
# corpus_freq_rank=1 means most frequent in corpus
# rank_ratio = corpus_rank / dat_rank
# If corpus_rank >> dat_rank: word is rare in corpus but common in DAT => overused in DAT
# If corpus_rank << dat_rank: word is common in corpus but rare in DAT => underused in DAT

# Filter to words with at least 10 occurrences for stable estimates
wdf_stable = wdf_ranked[wdf_ranked['dat_freq'] >= 10].copy()
print(f"Words with >= 10 DAT occurrences: {len(wdf_stable)}")

overused = wdf_stable.nlargest(20, 'rank_ratio')
underused = wdf_stable.nsmallest(20, 'rank_ratio')

# Scatter: dat_freq_rank vs corpus_freq_rank
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(np.log10(wdf_ranked['corpus_freq_rank']), np.log10(wdf_ranked['dat_freq_rank']),
           s=3, alpha=0.1, color='gray')

# Highlight overused (red) and underused (blue)
ax.scatter(np.log10(overused['corpus_freq_rank']), np.log10(overused['dat_freq_rank']),
           s=30, color='red', zorder=5, label='Overused (top 20)')
ax.scatter(np.log10(underused['corpus_freq_rank']), np.log10(underused['dat_freq_rank']),
           s=30, color='blue', zorder=5, label='Underused (top 20)')

# Diagonal
lim = [0, np.log10(max(wdf_ranked['corpus_freq_rank'].max(), wdf_ranked['dat_freq_rank'].max()))]
ax.plot(lim, lim, 'k--', alpha=0.5)

# Label top 10 each
for _, row in overused.head(10).iterrows():
    ax.annotate(row['word'], (np.log10(row['corpus_freq_rank']), np.log10(row['dat_freq_rank'])),
                fontsize=7, color='red', alpha=0.8)
for _, row in underused.head(10).iterrows():
    ax.annotate(row['word'], (np.log10(row['corpus_freq_rank']), np.log10(row['dat_freq_rank'])),
                fontsize=7, color='blue', alpha=0.8)

ax.set_xlabel('log10(Corpus Frequency Rank)', fontsize=12)
ax.set_ylabel('log10(DAT Frequency Rank)', fontsize=12)
ax.set_title('DAT vs Corpus Frequency Ranks\n(Above diagonal = overused in DAT)', fontsize=14)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(OUT / 'c1e_overused_underused.png', dpi=150)
plt.close()

# ─── 6. TABLES ───
print("\n" + "="*60)
print("TOP-10 TABLES")
print("="*60)

# Most frequent words in DAT
top_freq = wdf.nlargest(10, 'dat_freq')[['word', 'dat_freq', 'zipf_freq', 'mean_score_contrib']]
print("\n--- Top 10 Most Frequent Words in DAT ---")
print(top_freq.to_string(index=False))

# Highest scoring words (min 20 occurrences)
wdf_min20 = wdf[(wdf['dat_freq'] >= 20) & wdf['mean_score_contrib'].notna()]
top_score = wdf_min20.nlargest(10, 'mean_score_contrib')[['word', 'dat_freq', 'zipf_freq', 'mean_score_contrib']]
print("\n--- Top 10 Highest-Scoring Words (min 20 occurrences) ---")
print(top_score.to_string(index=False))

# Lowest scoring words (min 20 occurrences)
low_score = wdf_min20.nsmallest(10, 'mean_score_contrib')[['word', 'dat_freq', 'zipf_freq', 'mean_score_contrib']]
print("\n--- Top 10 Lowest-Scoring Words (min 20 occurrences) ---")
print(low_score.to_string(index=False))

# Most overused
print("\n--- Top 10 Most OVERUSED Words (high DAT freq relative to corpus) ---")
over_table = overused.head(10)[['word', 'dat_freq', 'corpus_freq_rank', 'dat_freq_rank', 'rank_ratio', 'zipf_freq']].copy()
over_table['rank_ratio'] = over_table['rank_ratio'].round(1)
print(over_table.to_string(index=False))

# Most underused
print("\n--- Top 10 Most UNDERUSED Words (low DAT freq relative to corpus) ---")
under_table = underused.head(10)[['word', 'dat_freq', 'corpus_freq_rank', 'dat_freq_rank', 'rank_ratio', 'zipf_freq']].copy()
under_table['rank_ratio'] = under_table['rank_ratio'].round(3)
print(under_table.to_string(index=False))

# ─── 7. Summary figure ───
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# 7a: corpus vs dat freq
ax = axes[0, 0]
mask = wdf_with_corpus['dat_freq'] > 0
x = np.log10(wdf_with_corpus.loc[mask, 'corpus_freq'])
y = np.log10(wdf_with_corpus.loc[mask, 'dat_freq'])
ax.scatter(x, y, alpha=0.08, s=3, color='steelblue')
slope, intercept, r, p, se = stats.linregress(x, y)
xline = np.linspace(x.min(), x.max(), 100)
ax.plot(xline, slope * xline + intercept, 'r-', linewidth=2)
ax.set_xlabel('log10(Corpus Freq)')
ax.set_ylabel('log10(DAT Freq)')
ax.set_title(f'A. Corpus vs DAT Frequency (r={r:.3f})')

# 7b: corpus freq vs score contrib
ax = axes[0, 1]
ax.scatter(wdf_score['log_corpus_freq'], wdf_score['mean_score_contrib'],
           alpha=0.08, s=3, color='coral')
slope2, intercept2, r2, p2, se2 = stats.linregress(wdf_score['log_corpus_freq'], wdf_score['mean_score_contrib'])
xline2 = np.linspace(wdf_score['log_corpus_freq'].min(), wdf_score['log_corpus_freq'].max(), 100)
ax.plot(xline2, slope2 * xline2 + intercept2, 'k-', linewidth=2)
ax.set_xlabel('log10(Corpus Freq)')
ax.set_ylabel('Mean Cosine Distance')
ax.set_title(f'B. Corpus Freq vs Score Contribution (r={r2:.3f})')

# 7c: avg rarity vs score
ax = axes[1, 0]
ax.scatter(df_valid['avg_log_corpus_freq'], df_valid['score'], alpha=0.03, s=3, color='purple')
slope3, intercept3, _, _, _ = stats.linregress(df_valid['avg_log_corpus_freq'], df_valid['score'])
xr = np.linspace(df_valid['avg_log_corpus_freq'].min(), df_valid['avg_log_corpus_freq'].max(), 100)
ax.plot(xr, slope3*xr + intercept3, 'r-', linewidth=2)
ax.set_xlabel('Mean log10(Corpus Freq) of 7 Words')
ax.set_ylabel('DAT Score')
ax.set_title(f'C. Word Rarity vs Score (r={r3:.3f})')

# 7d: Zipf
ax = axes[1, 1]
ax.scatter(np.log10(ranks), np.log10(dat_freqs), s=3, alpha=0.3, color='steelblue', label='DAT')
ax.scatter(np.log10(corpus_ranks), np.log10(corpus_f), s=3, alpha=0.3, color='coral', label='Corpus')
ax.set_xlabel('log10(Rank)')
ax.set_ylabel('log10(Frequency)')
ax.set_title('D. Zipf Plots: DAT vs Corpus')
ax.legend()

plt.tight_layout()
plt.savefig(OUT / 'c1_summary_4panel.png', dpi=150)
plt.close()

# Save word table
wdf.to_csv(OUT / 'c1_word_stats.csv', index=False)
print(f"\nWord stats saved to {OUT / 'c1_word_stats.csv'}")

# ─── 8. Additional stats ───
print("\n" + "="*60)
print("ADDITIONAL STATISTICS")
print("="*60)

# What fraction of words have zero corpus frequency?
n_zero = (wdf['corpus_freq'] == 0).sum()
print(f"Words with zero corpus frequency: {n_zero}/{len(wdf)} ({100*n_zero/len(wdf):.1f}%)")

# Mean Zipf frequency by DAT frequency quintile
wdf_nz = wdf[wdf['corpus_freq'] > 0].copy()
try:
    wdf_nz['dat_freq_quintile'] = pd.qcut(wdf_nz['dat_freq'], 5, duplicates='drop')
    print("\nMean Zipf frequency by DAT frequency quintile:")
    print(wdf_nz.groupby('dat_freq_quintile')['zipf_freq'].mean().to_string())
except Exception as e:
    # Fallback: use manual bins
    cuts = [0, 2, 5, 15, 50, wdf_nz['dat_freq'].max()+1]
    labels = ['1-2', '3-5', '6-15', '16-50', '51+']
    wdf_nz['dat_freq_bin'] = pd.cut(wdf_nz['dat_freq'], bins=cuts, labels=labels, right=False)
    print("\nMean Zipf frequency by DAT frequency bin:")
    print(wdf_nz.groupby('dat_freq_bin')['zipf_freq'].mean().to_string())

# Correlation: score vs avg_zipf
r4, p4 = stats.pearsonr(df_valid['avg_zipf'], df_valid['score'])
print(f"\nAvg Zipf freq vs score: r={r4:.4f}, p={p4:.2e}")

# Effect size: top vs bottom quartile of rarity
q25 = df_valid['avg_log_corpus_freq'].quantile(0.25)
q75 = df_valid['avg_log_corpus_freq'].quantile(0.75)
rare_scores = df_valid[df_valid['avg_log_corpus_freq'] <= q25]['score']
common_scores = df_valid[df_valid['avg_log_corpus_freq'] >= q75]['score']
d = (rare_scores.mean() - common_scores.mean()) / np.sqrt((rare_scores.std()**2 + common_scores.std()**2) / 2)
print(f"\nCohen's d (rare vs common word users): {d:.3f}")
print(f"  Rare word users (Q1) mean score: {rare_scores.mean():.2f}")
print(f"  Common word users (Q4) mean score: {common_scores.mean():.2f}")

print("\n=== C1 Analysis Complete ===")
