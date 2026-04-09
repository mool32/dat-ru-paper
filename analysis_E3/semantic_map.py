"""
E3: Semantic map visualization — UMAP of all words used in DAT dataset
"""
import pandas as pd
import numpy as np
import json
import struct
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
import umap
import warnings
warnings.filterwarnings('ignore')

OUT = '/Users/teo/Downloads/datcreativity/analysis_E3'
DATA = '/Users/teo/Downloads/datcreativity/data'

# ── 1. Load CSV and parse words ──
print("Loading CSV...")
df = pd.read_csv('/Users/teo/Downloads/DAT-RU Results - Sheet1 (1).csv')
print(f"  {len(df)} submissions")

# Parse words per submission
def parse_words(w):
    if pd.isna(w):
        return []
    return [x.strip().lower() for x in str(w).split(',') if x.strip()]

df['word_list'] = df['words'].apply(parse_words)

# Per-word statistics
word_scores = {}  # word -> list of scores
for _, row in df.iterrows():
    s = row['score']
    for w in row['word_list']:
        if w not in word_scores:
            word_scores[w] = []
        word_scores[w].append(s)

word_stats = pd.DataFrame({
    'word': list(word_scores.keys()),
    'dat_frequency': [len(v) for v in word_scores.values()],
    'mean_score': [np.mean(v) for v in word_scores.values()]
})
print(f"  {len(word_stats)} unique words")

# Load category map
with open('/Users/teo/Downloads/datcreativity/analysis_C2/category_map.json') as f:
    category_map = json.load(f)
word_stats['category'] = word_stats['word'].map(category_map).fillna('UNCATEGORIZED')

# ── 2. Load embeddings ──
print("Loading embeddings...")
with open(f'{DATA}/words.json') as f:
    vocab = json.load(f)
word_to_idx = {w: i for i, w in enumerate(vocab)}

# Read matrix header
with open(f'{DATA}/matrix.bin', 'rb') as f:
    num_words, dim = struct.unpack('II', f.read(8))
    print(f"  Embedding matrix: {num_words} x {dim}")
    raw = np.frombuffer(f.read(), dtype=np.int8).reshape(num_words, dim)

# Find words that are both in submissions and in embeddings
used_words = word_stats['word'].tolist()
found_words = []
found_indices = []
for w in used_words:
    if w in word_to_idx:
        found_words.append(w)
        found_indices.append(word_to_idx[w])

print(f"  {len(found_words)} / {len(used_words)} words found in embeddings")

# Build matrix
vectors = raw[found_indices].astype(np.float32)
# Normalize int8 to float
vectors = vectors / 127.0

# Build a lookup for found words
found_stats = word_stats[word_stats['word'].isin(set(found_words))].copy()
found_stats = found_stats.set_index('word').loc[found_words].reset_index()

# ── 3. Run UMAP ──
print("Running UMAP...")
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42, n_jobs=1)
embedding_2d = reducer.fit_transform(vectors)
found_stats['umap_x'] = embedding_2d[:, 0]
found_stats['umap_y'] = embedding_2d[:, 1]
print("  UMAP done.")

# ── Plot settings ──
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
})

def clean_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('UMAP 1', fontsize=11)
    ax.set_ylabel('UMAP 2', fontsize=11)

# ── 4a. Color by DAT frequency (log scale) ──
print("Plotting 4a: frequency map...")
fig, ax = plt.subplots(figsize=(12, 10))
freq = found_stats['dat_frequency'].values
log_freq = np.log10(freq + 1)
sizes = 1 + 15 * (log_freq / log_freq.max())

sc = ax.scatter(found_stats['umap_x'], found_stats['umap_y'],
                c=log_freq, cmap='viridis', s=sizes, alpha=0.6, edgecolors='none')
cbar = plt.colorbar(sc, ax=ax, shrink=0.7, label='log10(frequency + 1)')

# Annotate top 30 most frequent
top30 = found_stats.nlargest(30, 'dat_frequency')
for _, r in top30.iterrows():
    ax.annotate(r['word'], (r['umap_x'], r['umap_y']),
                fontsize=7, alpha=0.85, ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.7))

ax.set_title('Semantic Map: Word Frequency in DAT Submissions', fontsize=14, fontweight='bold')
clean_axes(ax)
plt.tight_layout()
plt.savefig(f'{OUT}/E3a_frequency_map.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved E3a_frequency_map.png")

# ── 4b. Color by mean score ──
print("Plotting 4b: mean score map...")
fig, ax = plt.subplots(figsize=(12, 10))
ms = found_stats['mean_score'].values
sizes_b = 2 + 10 * (np.log10(freq + 1) / log_freq.max())

sc = ax.scatter(found_stats['umap_x'], found_stats['umap_y'],
                c=ms, cmap='RdYlGn', s=sizes_b, alpha=0.5, edgecolors='none',
                vmin=np.percentile(ms, 5), vmax=np.percentile(ms, 95))
cbar = plt.colorbar(sc, ax=ax, shrink=0.7, label='Mean DAT Score')

# Annotate outliers: freq >= 20 with extreme scores
freq_mask = found_stats['dat_frequency'] >= 20
if freq_mask.sum() > 0:
    subset = found_stats[freq_mask]
    p10 = subset['mean_score'].quantile(0.05)
    p90 = subset['mean_score'].quantile(0.95)
    outliers = subset[(subset['mean_score'] <= p10) | (subset['mean_score'] >= p90)]
    for _, r in outliers.iterrows():
        color = 'darkred' if r['mean_score'] <= p10 else 'darkgreen'
        ax.annotate(r['word'], (r['umap_x'], r['umap_y']),
                    fontsize=7, alpha=0.85, color=color, ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.7))

ax.set_title('Semantic Map: Mean DAT Score per Word', fontsize=14, fontweight='bold')
clean_axes(ax)
plt.tight_layout()
plt.savefig(f'{OUT}/E3b_mean_score_map.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved E3b_mean_score_map.png")

# ── 4c. Color by semantic category ──
print("Plotting 4c: category map...")
categories = ['ANIMAL', 'FOOD', 'BODY', 'NATURE', 'OBJECT_HOME', 'OBJECT_TECH',
              'PLACE', 'ABSTRACT', 'SCIENCE', 'PERSON', 'ART_CULTURE', 'EMOTION', 'ACTION_PROCESS']
cat_colors = {
    'ANIMAL': '#e41a1c', 'FOOD': '#ff7f00', 'BODY': '#984ea3',
    'NATURE': '#4daf4a', 'OBJECT_HOME': '#a65628', 'OBJECT_TECH': '#377eb8',
    'PLACE': '#f781bf', 'ABSTRACT': '#999999', 'SCIENCE': '#00ced1',
    'PERSON': '#e6ab02', 'ART_CULTURE': '#66c2a5', 'EMOTION': '#fc8d62',
    'ACTION_PROCESS': '#8da0cb', 'UNCATEGORIZED': '#e0e0e0'
}

fig, ax = plt.subplots(figsize=(12, 10))
# Plot uncategorized first (background)
uncat = found_stats[found_stats['category'] == 'UNCATEGORIZED']
ax.scatter(uncat['umap_x'], uncat['umap_y'], c='#e0e0e0', s=2, alpha=0.3, edgecolors='none', zorder=1)

# Plot categorized words
for cat in categories:
    mask = found_stats['category'] == cat
    sub = found_stats[mask]
    if len(sub) > 0:
        ax.scatter(sub['umap_x'], sub['umap_y'], c=cat_colors[cat], s=12, alpha=0.7,
                   edgecolors='none', label=cat, zorder=2)

ax.legend(loc='upper left', fontsize=7, ncol=2, framealpha=0.8, markerscale=2)
ax.set_title('Semantic Map: Category Distribution', fontsize=14, fontweight='bold')
clean_axes(ax)
plt.tight_layout()
plt.savefig(f'{OUT}/E3c_category_map.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved E3c_category_map.png")

# ── 4d. Combined insight map ──
print("Plotting 4d: combined insight map...")
fig, ax = plt.subplots(figsize=(14, 11))

# Background: all words as small gray dots
ax.scatter(found_stats['umap_x'], found_stats['umap_y'],
           c='#d0d0d0', s=1, alpha=0.2, edgecolors='none', zorder=1)

# Overlay: top 200 most frequent words
top200 = found_stats.nlargest(200, 'dat_frequency').copy()
t200_freq = np.log10(top200['dat_frequency'].values + 1)
t200_sizes = 20 + 200 * (t200_freq / t200_freq.max())
t200_scores = top200['mean_score'].values

sc = ax.scatter(top200['umap_x'], top200['umap_y'],
                c=t200_scores, cmap='RdYlGn', s=t200_sizes, alpha=0.8,
                edgecolors='black', linewidths=0.3, zorder=3,
                vmin=np.percentile(found_stats['mean_score'], 10),
                vmax=np.percentile(found_stats['mean_score'], 90))
cbar = plt.colorbar(sc, ax=ax, shrink=0.6, label='Mean DAT Score')

# Annotate top 20 most common
top20 = found_stats.nlargest(20, 'dat_frequency')
for _, r in top20.iterrows():
    ax.annotate(r['word'], (r['umap_x'], r['umap_y']),
                fontsize=8, fontweight='bold', ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='gray', alpha=0.85),
                zorder=5)

# Annotate top 10 highest-mean-score words (freq >= 10 to avoid noise)
high_score = found_stats[found_stats['dat_frequency'] >= 10].nlargest(10, 'mean_score')
already_labeled = set(top20['word'].values)
for _, r in high_score.iterrows():
    if r['word'] not in already_labeled:
        ax.annotate(r['word'], (r['umap_x'], r['umap_y']),
                    fontsize=7, fontstyle='italic', color='darkgreen', ha='center', va='top',
                    bbox=dict(boxstyle='round,pad=0.15', fc='#e8ffe8', ec='green', alpha=0.8),
                    zorder=5)

ax.set_title('Semantic Map: Overcrowded Zones vs Creative Territories', fontsize=14, fontweight='bold')
clean_axes(ax)
plt.tight_layout()
plt.savefig(f'{OUT}/E3d_combined_insight_map.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved E3d_combined_insight_map.png")

# ── 5. Additional analysis ──
print("Running additional analyses...")

# 5a. 2D KDE density of word usage
print("  Computing KDE...")
# Weight by frequency for density
x = found_stats['umap_x'].values
y = found_stats['umap_y'].values
freq_weights = found_stats['dat_frequency'].values

# Sample points proportional to frequency for KDE (cap to avoid memory issues)
max_samples = 100000
total_freq = freq_weights.sum()
probs = freq_weights / total_freq
n_samples = min(max_samples, total_freq)
sampled_idx = np.random.choice(len(x), size=int(n_samples), p=probs, replace=True)
kde_x = x[sampled_idx]
kde_y = y[sampled_idx]

kde = gaussian_kde(np.vstack([kde_x, kde_y]), bw_method=0.05)
# Evaluate on grid
xmin, xmax = x.min() - 1, x.max() + 1
ymin, ymax = y.min() - 1, y.max() + 1
xx, yy = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
positions = np.vstack([xx.ravel(), yy.ravel()])
density = kde(positions).reshape(xx.shape)

fig, ax = plt.subplots(figsize=(12, 10))
ax.contourf(xx, yy, density, levels=30, cmap='hot_r', alpha=0.8)
ax.contour(xx, yy, density, levels=10, colors='black', linewidths=0.3, alpha=0.4)
ax.scatter(x, y, c='white', s=0.5, alpha=0.1, edgecolors='none')

# Mark "empty zones" — low density areas
threshold = np.percentile(density, 10)
# Annotate top 15 most frequent words on density map
for _, r in top30.head(15).iterrows():
    ax.annotate(r['word'], (r['umap_x'], r['umap_y']),
                fontsize=7, color='white', fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.15', fc='black', ec='none', alpha=0.6))

ax.set_title('Semantic Space Density: Where People Go', fontsize=14, fontweight='bold')
clean_axes(ax)
plt.tight_layout()
plt.savefig(f'{OUT}/E3e_density_kde.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved E3e_density_kde.png")

# 5b. Top 10% vs Bottom 10% scorers comparison
print("  Computing top vs bottom scorer comparison...")
p10 = df['score'].quantile(0.10)
p90 = df['score'].quantile(0.90)
top_submissions = df[df['score'] >= p90]
bottom_submissions = df[df['score'] <= p10]

# Collect words for each group
def collect_words(sub_df):
    words = {}
    for wlist in sub_df['word_list']:
        for w in wlist:
            words[w] = words.get(w, 0) + 1
    return words

top_words = collect_words(top_submissions)
bottom_words = collect_words(bottom_submissions)

# Map to UMAP coordinates
word_to_umap = dict(zip(found_stats['word'], zip(found_stats['umap_x'], found_stats['umap_y'])))

fig, axes = plt.subplots(1, 2, figsize=(22, 10))

for ax, words_dict, title, color in [
    (axes[0], bottom_words, 'Bottom 10% Scorers', '#d62728'),
    (axes[1], top_words, 'Top 10% Scorers', '#2ca02c')
]:
    # Background
    ax.scatter(found_stats['umap_x'], found_stats['umap_y'],
               c='#e8e8e8', s=1, alpha=0.15, edgecolors='none')

    # Overlay group's words
    gx, gy, gf = [], [], []
    for w, f in words_dict.items():
        if w in word_to_umap:
            ux, uy = word_to_umap[w]
            gx.append(ux)
            gy.append(uy)
            gf.append(f)
    gx, gy, gf = np.array(gx), np.array(gy), np.array(gf)
    log_gf = np.log10(gf + 1)
    sizes = 3 + 50 * (log_gf / log_gf.max())

    ax.scatter(gx, gy, c=color, s=sizes, alpha=0.5, edgecolors='none')

    # Annotate top 15 for this group
    top_in_group = sorted(words_dict.items(), key=lambda x: -x[1])[:15]
    for w, f in top_in_group:
        if w in word_to_umap:
            ux, uy = word_to_umap[w]
            ax.annotate(w, (ux, uy), fontsize=7, fontweight='bold',
                       ha='center', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.15', fc='white', ec=color, alpha=0.8))

    ax.set_title(f'{title}\n({len(words_dict)} unique words)', fontsize=13, fontweight='bold')
    clean_axes(ax)

plt.suptitle('Semantic Territory: Top vs Bottom Scorers', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{OUT}/E3f_top_vs_bottom.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved E3f_top_vs_bottom.png")

# ── Save UMAP coordinates for later use ──
found_stats.to_csv(f'{OUT}/umap_word_data.csv', index=False)
print(f"\nSaved UMAP data for {len(found_stats)} words to umap_word_data.csv")

# ── Print summary stats ──
print("\n=== SUMMARY ===")
print(f"Total unique words in submissions: {len(word_stats)}")
print(f"Words with embeddings: {len(found_words)}")
print(f"Words with category labels: {(found_stats['category'] != 'UNCATEGORIZED').sum()}")
print(f"\nTop 10 most frequent words:")
for _, r in found_stats.nlargest(10, 'dat_frequency').iterrows():
    print(f"  {r['word']:20s}  freq={int(r['dat_frequency']):5d}  mean_score={r['mean_score']:.1f}  cat={r['category']}")
print(f"\nTop 10 highest mean_score words (freq >= 20):")
hs = found_stats[found_stats['dat_frequency'] >= 20].nlargest(10, 'mean_score')
for _, r in hs.iterrows():
    print(f"  {r['word']:20s}  freq={int(r['dat_frequency']):5d}  mean_score={r['mean_score']:.1f}  cat={r['category']}")
print(f"\nBottom 10 lowest mean_score words (freq >= 20):")
ls = found_stats[found_stats['dat_frequency'] >= 20].nsmallest(10, 'mean_score')
for _, r in ls.iterrows():
    print(f"  {r['word']:20s}  freq={int(r['dat_frequency']):5d}  mean_score={r['mean_score']:.1f}  cat={r['category']}")

# Top vs bottom territory stats
top_cats = found_stats[found_stats['word'].isin(top_words)]['category'].value_counts()
bot_cats = found_stats[found_stats['word'].isin(bottom_words)]['category'].value_counts()
print(f"\nTop 10% scorers - category distribution:")
for cat, cnt in top_cats.head(8).items():
    print(f"  {cat:20s}  {cnt}")
print(f"\nBottom 10% scorers - category distribution:")
for cat, cnt in bot_cats.head(8).items():
    print(f"  {cat:20s}  {cnt}")

# Coverage stats
print(f"\nTop 10% unique words: {len(top_words)}")
print(f"Bottom 10% unique words: {len(bottom_words)}")
print(f"Overlap: {len(set(top_words) & set(bottom_words))}")
print(f"Only in top: {len(set(top_words) - set(bottom_words))}")
print(f"Only in bottom: {len(set(bottom_words) - set(top_words))}")
