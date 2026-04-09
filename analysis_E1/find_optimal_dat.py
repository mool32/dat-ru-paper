#!/usr/bin/env python3
"""
E1: Find the theoretical maximum DAT score from 68K Russian lemmas.
"""

import json
import struct
import numpy as np
import time
import os

OUT_DIR = "/Users/teo/Downloads/datcreativity/analysis_E1"
DATA_DIR = "/Users/teo/Downloads/datcreativity/data"

# ── 1. Load data ──────────────────────────────────────────────────────────
print("Loading words.json ...")
with open(os.path.join(DATA_DIR, "words.json"), "r", encoding="utf-8") as f:
    words = json.load(f)
N = len(words)
print(f"  {N} words loaded")

print("Loading matrix.bin ...")
with open(os.path.join(DATA_DIR, "matrix.bin"), "rb") as f:
    num_words, dim = struct.unpack("II", f.read(8))
    print(f"  Header: {num_words} words × {dim} dims")
    raw = np.frombuffer(f.read(), dtype=np.int8).reshape(num_words, dim)

# Convert to float32 and L2-normalize
print("Normalizing to unit vectors ...")
matrix = raw.astype(np.float32)
norms = np.linalg.norm(matrix, axis=1, keepdims=True)
norms[norms == 0] = 1.0  # avoid division by zero
matrix /= norms
del raw, norms
print(f"  Matrix shape: {matrix.shape}, dtype: {matrix.dtype}")

# ── Helper: compute DAT score for a set of word indices ───────────────────
def compute_score(indices):
    """Compute raw DAT score (mean cosine distance × 100) for a set of word indices."""
    vecs = matrix[indices]  # (k, 300)
    # cosine similarity matrix
    sim = vecs @ vecs.T  # (k, k)
    k = len(indices)
    # extract upper triangle (all pairs)
    triu_idx = np.triu_indices(k, k=1)
    sims = sim[triu_idx]
    dists = 1.0 - sims  # cosine distance
    return float(np.mean(dists)) * 100.0

def adjusted_score(raw):
    """Convert raw score to adjusted score."""
    return 47.4548 * (raw / 100.0) ** 3.3820 + 48.5157

# ── 2. Greedy algorithm ──────────────────────────────────────────────────
print("\n=== GREEDY ALGORITHM ===")

def greedy_search(start_pair=None):
    """
    Greedy search for 7 words maximizing mean pairwise cosine distance.
    If start_pair is None, find the globally best pair first.
    """
    if start_pair is None:
        # Find the pair with maximum cosine distance
        # We can't compute full NxN, but we can find most distant pair
        # Strategy: cosine distance = 1 - cos_sim. Max distance = 1 - min_sim.
        # Most distant pair has most negative dot product (for unit vectors).
        # Use random sampling to find good candidates, then refine.
        print("  Finding best starting pair ...")

        # For the absolute best pair, we need words pointing in opposite directions.
        # Sample approach: pick random vectors, find their most distant partner.
        best_dist = -1
        best_pair = None

        # Strategy: for each word, the most distant word has the most negative dot product.
        # We can compute this in batches.
        batch_size = 5000
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            # (batch, 300) @ (300, N) -> (batch, N) — similarities
            sims = matrix[start:end] @ matrix.T  # (batch, N)
            # For each row, find minimum similarity (= max distance)
            min_sims = np.min(sims, axis=1)
            min_idx = np.argmin(sims, axis=1)
            # Find best in this batch
            best_in_batch = np.argmin(min_sims)
            if min_sims[best_in_batch] < -best_dist + 1:  # dist = 1 - sim
                dist = 1.0 - min_sims[best_in_batch]
                if dist > best_dist:
                    best_dist = dist
                    best_pair = (start + best_in_batch, min_idx[best_in_batch])
            if start % 20000 == 0:
                print(f"    Scanned {start}/{N} words, best dist so far: {best_dist:.4f}")

        selected = list(best_pair)
        print(f"  Best pair: '{words[selected[0]]}' & '{words[selected[1]]}', dist={best_dist:.4f}")
    else:
        selected = list(start_pair)

    # Greedily add words 3-7
    for step in range(len(selected), 7):
        sel_vecs = matrix[selected]  # (k, 300)
        # Compute similarity of all N words to each selected word
        # sims: (N, k)
        sims = matrix @ sel_vecs.T
        # For each candidate, compute mean distance to all selected words
        # Plus we need pairwise distances among selected (already fixed)
        # Total pairs when we add candidate: C(k+1, 2) = C(k,2) + k
        # Current sum of distances among selected
        k = len(selected)
        current_pairs = k * (k - 1) // 2
        current_sum = 0.0
        for i in range(k):
            for j in range(i + 1, k):
                current_sum += 1.0 - float(sel_vecs[i] @ sel_vecs[j])

        # For each candidate: new_sum = current_sum + sum of distances to each selected word
        # new_sum = current_sum + sum(1 - sim[candidate, sel_i]) for each sel_i
        # = current_sum + k - sum(sim[candidate, sel_i])
        new_pair_count = current_pairs + k
        candidate_sums = current_sum + k - np.sum(sims, axis=1)  # (N,)
        candidate_means = candidate_sums / new_pair_count

        # Exclude already selected
        for idx in selected:
            candidate_means[idx] = -1.0

        best_idx = int(np.argmax(candidate_means))
        selected.append(best_idx)
        score = candidate_means[best_idx] * 100.0
        print(f"  Added word {step+1}: '{words[best_idx]}' (running mean dist×100 = {score:.2f})")

    raw = compute_score(selected)
    adj = adjusted_score(raw)
    return selected, raw, adj

t0 = time.time()
greedy_indices, greedy_raw, greedy_adj = greedy_search()
t1 = time.time()
print(f"\nGreedy result ({t1-t0:.1f}s):")
print(f"  Words: {[words[i] for i in greedy_indices]}")
print(f"  Raw score: {greedy_raw:.2f}")
print(f"  Adjusted score: {greedy_adj:.2f}")

# ── 3. Simulated annealing ───────────────────────────────────────────────
print("\n=== SIMULATED ANNEALING ===")

def simulated_annealing(initial_indices, n_iter=200_000, T_start=0.5, T_end=0.001):
    """SA to improve a 7-word solution."""
    current = list(initial_indices)
    current_score = compute_score(current)
    best = list(current)
    best_score = current_score

    rng = np.random.default_rng(42)
    current_set = set(current)

    T = T_start
    cooling = (T_end / T_start) ** (1.0 / n_iter)

    accepted = 0
    for it in range(n_iter):
        # Pick random position to replace
        pos = rng.integers(0, 7)
        old_idx = current[pos]

        # Pick random replacement (not in current set)
        while True:
            new_idx = rng.integers(0, N)
            if new_idx not in current_set:
                break

        # Compute new score
        trial = list(current)
        trial[pos] = new_idx
        trial_score = compute_score(trial)

        delta = trial_score - current_score
        if delta > 0 or rng.random() < np.exp(delta / T):
            current_set.discard(old_idx)
            current_set.add(new_idx)
            current = trial
            current_score = trial_score
            accepted += 1

            if current_score > best_score:
                best = list(current)
                best_score = current_score

        T *= cooling

        if (it + 1) % 50000 == 0:
            print(f"  Iter {it+1}/{n_iter}: T={T:.4f}, current={current_score:.4f}, "
                  f"best={best_score:.4f}, accepted={accepted}")
            accepted = 0

    return best, best_score

t0 = time.time()
sa_indices, sa_raw = simulated_annealing(greedy_indices)
sa_adj = adjusted_score(sa_raw)
t1 = time.time()
print(f"\nSA result ({t1-t0:.1f}s):")
print(f"  Words: {[words[i] for i in sa_indices]}")
print(f"  Raw score: {sa_raw:.2f}")
print(f"  Adjusted score: {sa_adj:.2f}")

# ── 4. Random restarts (100 greedy runs from random pairs) ────────────────
print("\n=== RANDOM RESTART GREEDY (100 runs) ===")

def greedy_from_random_pair(rng):
    """One greedy run starting from a random pair."""
    # Pick two random words
    pair = rng.choice(N, size=2, replace=False).tolist()
    selected = pair

    for step in range(2, 7):
        sel_vecs = matrix[selected]
        sims = matrix @ sel_vecs.T
        k = len(selected)
        current_pairs = k * (k - 1) // 2
        current_sum = 0.0
        for i in range(k):
            for j in range(i + 1, k):
                current_sum += 1.0 - float(sel_vecs[i] @ sel_vecs[j])

        new_pair_count = current_pairs + k
        candidate_sums = current_sum + k - np.sum(sims, axis=1)
        for idx in selected:
            candidate_sums[idx] = -999.0 * new_pair_count
        candidate_means = candidate_sums / new_pair_count

        best_idx = int(np.argmax(candidate_means))
        selected.append(best_idx)

    return selected, compute_score(selected)

rng = np.random.default_rng(123)
best_rr_indices = None
best_rr_raw = -1
t0 = time.time()

for run in range(100):
    indices, raw = greedy_from_random_pair(rng)
    if raw > best_rr_raw:
        best_rr_raw = raw
        best_rr_indices = indices
    if (run + 1) % 20 == 0:
        print(f"  Run {run+1}/100, best so far: {best_rr_raw:.4f}")

best_rr_adj = adjusted_score(best_rr_raw)
t1 = time.time()
print(f"\nRandom restart best ({t1-t0:.1f}s):")
print(f"  Words: {[words[i] for i in best_rr_indices]}")
print(f"  Raw score: {best_rr_raw:.2f}")
print(f"  Adjusted score: {best_rr_adj:.2f}")

# ── Run SA on the best random restart too ─────────────────────────────────
print("\n=== SA on random restart best ===")
t0 = time.time()
sa2_indices, sa2_raw = simulated_annealing(best_rr_indices, n_iter=200_000)
sa2_adj = adjusted_score(sa2_raw)
t1 = time.time()
print(f"\nSA2 result ({t1-t0:.1f}s):")
print(f"  Words: {[words[i] for i in sa2_indices]}")
print(f"  Raw score: {sa2_raw:.2f}")
print(f"  Adjusted score: {sa2_adj:.2f}")

# ── Pick overall best ────────────────────────────────────────────────────
candidates = [
    ("Greedy", greedy_indices, greedy_raw, greedy_adj),
    ("SA on Greedy", sa_indices, sa_raw, sa_adj),
    ("Random Restart", best_rr_indices, best_rr_raw, best_rr_adj),
    ("SA on RR", sa2_indices, sa2_raw, sa2_adj),
]
candidates.sort(key=lambda x: x[2], reverse=True)
best_method, best_indices, best_raw, best_adj = candidates[0]
print(f"\n*** OVERALL BEST: {best_method} ***")
print(f"  Words: {[words[i] for i in best_indices]}")
print(f"  Raw: {best_raw:.4f}, Adjusted: {best_adj:.4f}")

# ── 5. Random baseline (10K sets) ────────────────────────────────────────
print("\n=== RANDOM BASELINE (10,000 random 7-word sets) ===")
rng2 = np.random.default_rng(456)
random_scores = []
t0 = time.time()
for i in range(10_000):
    idx = rng2.choice(N, size=7, replace=False).tolist()
    random_scores.append(compute_score(idx))
    if (i + 1) % 2000 == 0:
        print(f"  {i+1}/10000 ...")
random_scores = np.array(random_scores)
random_adj = np.array([adjusted_score(r) for r in random_scores])
t1 = time.time()
print(f"Random baseline ({t1-t0:.1f}s):")
print(f"  Raw  — mean: {random_scores.mean():.2f}, std: {random_scores.std():.2f}, "
      f"min: {random_scores.min():.2f}, max: {random_scores.max():.2f}")
print(f"  Adj  — mean: {random_adj.mean():.2f}, std: {random_adj.std():.2f}, "
      f"min: {random_adj.min():.2f}, max: {random_adj.max():.2f}")

# ── 6. Best with common words only (top 500 by index = most frequent) ────
print("\n=== BEST WITH COMMON WORDS (top 500 by frequency) ===")
# Assuming words are sorted by frequency (most common first)
common_indices = list(range(500))
# Greedy within common words
best_common_pair = None
best_common_dist = -1
common_matrix = matrix[:500]  # (500, 300)
common_sims = common_matrix @ common_matrix.T  # (500, 500) — small enough
common_dists = 1.0 - common_sims

# Find best pair
for i in range(500):
    for j in range(i + 1, 500):
        if common_dists[i, j] > best_common_dist:
            best_common_dist = common_dists[i, j]
            best_common_pair = [i, j]

# Greedy from best pair within common words
selected_common = list(best_common_pair)
for step in range(2, 7):
    k = len(selected_common)
    current_pairs = k * (k - 1) // 2
    current_sum = sum(common_dists[selected_common[i], selected_common[j]]
                      for i in range(k) for j in range(i + 1, k))

    best_mean = -1
    best_idx = -1
    new_pair_count = current_pairs + k
    for c in range(500):
        if c in selected_common:
            continue
        new_sum = current_sum + sum(common_dists[c, s] for s in selected_common)
        mean = new_sum / new_pair_count
        if mean > best_mean:
            best_mean = mean
            best_idx = c
    selected_common.append(best_idx)

common_raw = compute_score(selected_common)
common_adj = adjusted_score(common_raw)
print(f"  Words: {[words[i] for i in selected_common]}")
print(f"  Raw: {common_raw:.2f}, Adjusted: {common_adj:.2f}")

# ── 7. Percentile calculations ───────────────────────────────────────────
print("\n=== PERCENTILE ANALYSIS ===")
mean_human_raw = 87.9
best_human_raw = 104.8
# Back-compute raw from adjusted using the formula: adjusted = 47.4548*(raw/100)^3.382 + 48.5157
# For percentile, we'll express as % of ceiling
mean_human_pct = (mean_human_raw / best_raw) * 100
best_human_pct = (best_human_raw / best_raw) * 100
print(f"  Theoretical ceiling (raw): {best_raw:.2f}")
print(f"  Mean human ({mean_human_raw}) = {mean_human_pct:.1f}% of ceiling")
print(f"  Best human ({best_human_raw}) = {best_human_pct:.1f}% of ceiling")
print(f"  Common-word ceiling (raw): {common_raw:.2f}")
common_pct = (common_raw / best_raw) * 100
print(f"  Common-word ceiling = {common_pct:.1f}% of theoretical ceiling")

# ── 8. Save all results ──────────────────────────────────────────────────
results = {
    "theoretical_ceiling": {
        "method": best_method,
        "word_indices": best_indices,
        "words": [words[i] for i in best_indices],
        "raw_score": round(best_raw, 4),
        "adjusted_score": round(best_adj, 4),
    },
    "greedy": {
        "words": [words[i] for i in greedy_indices],
        "raw": round(greedy_raw, 4),
        "adjusted": round(greedy_adj, 4),
    },
    "sa_on_greedy": {
        "words": [words[i] for i in sa_indices],
        "raw": round(sa_raw, 4),
        "adjusted": round(sa_adj, 4),
    },
    "random_restart_best": {
        "words": [words[i] for i in best_rr_indices],
        "raw": round(best_rr_raw, 4),
        "adjusted": round(best_rr_adj, 4),
    },
    "sa_on_rr": {
        "words": [words[i] for i in sa2_indices],
        "raw": round(sa2_raw, 4),
        "adjusted": round(sa2_adj, 4),
    },
    "common_words_ceiling": {
        "words": [words[i] for i in selected_common],
        "raw": round(common_raw, 4),
        "adjusted": round(common_adj, 4),
    },
    "random_baseline": {
        "n": 10000,
        "raw_mean": round(float(random_scores.mean()), 2),
        "raw_std": round(float(random_scores.std()), 2),
        "raw_max": round(float(random_scores.max()), 2),
        "adj_mean": round(float(random_adj.mean()), 2),
        "adj_std": round(float(random_adj.std()), 2),
        "adj_max": round(float(random_adj.max()), 2),
    },
    "percentiles": {
        "mean_human_pct_of_ceiling": round(mean_human_pct, 1),
        "best_human_pct_of_ceiling": round(best_human_pct, 1),
        "common_pct_of_ceiling": round(common_pct, 1),
    },
    "all_methods_sorted": [(m, round(r, 4), round(a, 4)) for m, _, r, a in candidates],
}

with open(os.path.join(OUT_DIR, "results_E1.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\nResults saved to {OUT_DIR}/results_E1.json")

# Save scores for plotting
np.save(os.path.join(OUT_DIR, "random_raw_scores.npy"), random_scores)
np.save(os.path.join(OUT_DIR, "random_adj_scores.npy"), random_adj)
print("Random scores saved for plotting.")

print("\n=== DONE ===")
