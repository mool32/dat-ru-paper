#!/usr/bin/env python3
"""
More aggressive SA: higher temperature, more iterations, multiple restarts.
"""
import json, struct, numpy as np, time, os

OUT_DIR = "/Users/teo/Downloads/datcreativity/analysis_E1"
DATA_DIR = "/Users/teo/Downloads/datcreativity/data"

# Load
with open(os.path.join(DATA_DIR, "words.json"), "r", encoding="utf-8") as f:
    words = json.load(f)
N = len(words)

with open(os.path.join(DATA_DIR, "matrix.bin"), "rb") as f:
    num_words, dim = struct.unpack("II", f.read(8))
    raw = np.frombuffer(f.read(), dtype=np.int8).reshape(num_words, dim)
matrix = raw.astype(np.float32)
norms = np.linalg.norm(matrix, axis=1, keepdims=True)
norms[norms == 0] = 1.0
matrix /= norms
del raw, norms

def compute_score(indices):
    vecs = matrix[indices]
    sim = vecs @ vecs.T
    k = len(indices)
    triu_idx = np.triu_indices(k, k=1)
    return float(np.mean(1.0 - sim[triu_idx])) * 100.0

def adjusted_score(raw):
    return 47.4548 * (raw / 100.0) ** 3.3820 + 48.5157

# Efficient SA: precompute pairwise distances incrementally
def sa_fast(initial_indices, n_iter=500_000, T_start=2.0, T_end=0.0005, seed=42):
    rng = np.random.default_rng(seed)
    current = np.array(initial_indices, dtype=np.int64)

    # Precompute current pairwise similarities
    vecs = matrix[current]  # (7, 300)
    # Store pairwise sims as a flat array for the 21 pairs
    k = 7
    pair_sims = np.zeros(21, dtype=np.float32)
    idx = 0
    for i in range(k):
        for j in range(i+1, k):
            pair_sims[idx] = float(vecs[i] @ vecs[j])
            idx += 1

    current_mean_dist = float(np.mean(1.0 - pair_sims))
    best = current.copy()
    best_score = current_mean_dist * 100.0

    T = T_start
    cooling = (T_end / T_start) ** (1.0 / n_iter)

    current_set = set(current.tolist())
    improved = 0

    for it in range(n_iter):
        pos = rng.integers(0, 7)
        old_idx = int(current[pos])

        while True:
            new_idx = int(rng.integers(0, N))
            if new_idx not in current_set:
                break

        # Compute new pairwise sims efficiently:
        # Only pairs involving position 'pos' change
        new_vec = matrix[new_idx]
        new_pair_sims = pair_sims.copy()
        idx = 0
        for i in range(k):
            for j in range(i+1, k):
                if i == pos or j == pos:
                    other = j if i == pos else i
                    new_pair_sims[idx] = float(new_vec @ matrix[current[other]])
                idx += 1

        new_mean_dist = float(np.mean(1.0 - new_pair_sims))
        delta = new_mean_dist - current_mean_dist

        if delta > 0 or rng.random() < np.exp(delta * 100.0 / T):
            current_set.discard(old_idx)
            current_set.add(new_idx)
            current[pos] = new_idx
            pair_sims = new_pair_sims
            current_mean_dist = new_mean_dist

            score = current_mean_dist * 100.0
            if score > best_score:
                best = current.copy()
                best_score = score
                improved += 1

        T *= cooling

        if (it + 1) % 100000 == 0:
            print(f"  Iter {it+1}/{n_iter}: T={T:.5f}, best={best_score:.4f}, improved={improved}")

    return best.tolist(), best_score

# Start from the best known solution
best_known = None
best_known_score = 0

# Run multiple SA restarts
seeds_and_starts = []

# Load previous best
with open(os.path.join(OUT_DIR, "results_E1.json"), "r") as f:
    prev = json.load(f)

# Get word indices for previous bests
def get_indices(word_list):
    word_to_idx = {w: i for i, w in enumerate(words)}
    return [word_to_idx[w] for w in word_list]

starts = [
    ("greedy", get_indices(prev["greedy"]["words"])),
    ("rr_best", get_indices(prev["random_restart_best"]["words"])),
]

for name, start_idx in starts:
    for seed in [42, 137, 999, 2024, 7777]:
        print(f"\nSA: {name}, seed={seed}")
        t0 = time.time()
        idx, score = sa_fast(start_idx, n_iter=500_000, T_start=3.0, T_end=0.0001, seed=seed)
        t1 = time.time()
        print(f"  Result: {score:.4f} ({t1-t0:.1f}s) — {[words[i] for i in idx]}")
        if score > best_known_score:
            best_known = idx
            best_known_score = score

print(f"\n*** BEST FROM AGGRESSIVE SA ***")
print(f"  Words: {[words[i] for i in best_known]}")
print(f"  Raw: {best_known_score:.4f}")
print(f"  Adjusted: {adjusted_score(best_known_score):.4f}")

# Update results
prev["aggressive_sa"] = {
    "words": [words[i] for i in best_known],
    "raw": round(best_known_score, 4),
    "adjusted": round(adjusted_score(best_known_score), 4),
}

# Update ceiling if improved
if best_known_score > prev["theoretical_ceiling"]["raw_score"]:
    prev["theoretical_ceiling"] = {
        "method": "Aggressive SA",
        "word_indices": best_known,
        "words": [words[i] for i in best_known],
        "raw_score": round(best_known_score, 4),
        "adjusted_score": round(adjusted_score(best_known_score), 4),
    }
    # Recompute percentiles
    prev["percentiles"]["mean_human_pct_of_ceiling"] = round(87.9 / best_known_score * 100, 1)
    prev["percentiles"]["best_human_pct_of_ceiling"] = round(104.8 / best_known_score * 100, 1)

with open(os.path.join(OUT_DIR, "results_E1.json"), "w", encoding="utf-8") as f:
    json.dump(prev, f, ensure_ascii=False, indent=2)

print("Results updated.")
