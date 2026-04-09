#!/usr/bin/env python3
"""
E1 Plot: Distribution of random 7-word DAT scores vs human benchmarks and theoretical ceiling.
Also: common-words analysis using a curated list of frequent Russian words.
"""
import json, struct, numpy as np, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "/Users/teo/Downloads/datcreativity/analysis_E1"
DATA_DIR = "/Users/teo/Downloads/datcreativity/data"

# Load results
with open(os.path.join(OUT_DIR, "results_E1.json")) as f:
    results = json.load(f)

random_raw = np.load(os.path.join(OUT_DIR, "random_raw_scores.npy"))
random_adj = np.load(os.path.join(OUT_DIR, "random_adj_scores.npy"))

# ── Common words analysis ─────────────────────────────────────────────────
# Words.json is alphabetically sorted (not by frequency).
# We'll identify common Russian words from the vocabulary and find the best set among them.
with open(os.path.join(DATA_DIR, "words.json"), "r", encoding="utf-8") as f:
    words = json.load(f)
N = len(words)

with open(os.path.join(DATA_DIR, "matrix.bin"), "rb") as f:
    num_words, dim = struct.unpack("II", f.read(8))
    raw_data = np.frombuffer(f.read(), dtype=np.int8).reshape(num_words, dim)
matrix = raw_data.astype(np.float32)
norms = np.linalg.norm(matrix, axis=1, keepdims=True)
norms[norms == 0] = 1.0
matrix /= norms
del raw_data, norms

def compute_score(indices):
    vecs = matrix[indices]
    sim = vecs @ vecs.T
    k = len(indices)
    triu_idx = np.triu_indices(k, k=1)
    return float(np.mean(1.0 - sim[triu_idx])) * 100.0

def adjusted_score(raw):
    return 47.4548 * (raw / 100.0) ** 3.3820 + 48.5157

# Curated list of ~200 very common Russian words (nouns, verbs, adjectives)
common_russian = [
    "дом", "вода", "огонь", "земля", "небо", "солнце", "луна", "звезда",
    "дерево", "цветок", "птица", "рыба", "кошка", "собака", "лошадь",
    "человек", "ребенок", "мужчина", "женщина", "друг", "враг",
    "книга", "стол", "стул", "окно", "дверь", "стена", "пол", "потолок",
    "город", "деревня", "улица", "дорога", "мост", "река", "море", "озеро",
    "гора", "лес", "поле", "сад", "парк", "школа", "больница",
    "машина", "поезд", "самолет", "корабль", "велосипед",
    "хлеб", "молоко", "мясо", "рис", "сахар", "соль",
    "время", "день", "ночь", "утро", "вечер", "год", "месяц", "неделя",
    "работа", "игра", "песня", "танец", "музыка", "картина",
    "любовь", "счастье", "страх", "боль", "радость", "грусть",
    "война", "мир", "сила", "правда", "ложь", "закон",
    "красный", "синий", "зеленый", "белый", "черный", "желтый",
    "большой", "маленький", "старый", "новый", "быстрый", "медленный",
    "камень", "железо", "золото", "серебро", "стекло", "бумага",
    "нож", "ключ", "часы", "зеркало", "лампа", "телефон",
    "снег", "дождь", "ветер", "облако", "гроза", "туман",
    "молоток", "колесо", "веревка", "ткань", "кирпич", "песок",
    "яблоко", "хлеб", "масло", "сыр", "чай", "кофе",
    "палец", "рука", "нога", "голова", "глаз", "ухо", "нос", "рот",
    "сердце", "кровь", "кость", "кожа", "волос", "зуб",
    "письмо", "газета", "журнал", "карта", "билет", "деньги",
    "церковь", "театр", "магазин", "ресторан", "музей", "библиотека",
    "полиция", "армия", "флаг", "граница", "столица", "государство",
    "медведь", "волк", "лиса", "заяц", "олень", "змея", "корова",
    "пчела", "бабочка", "паук", "муравей", "червь",
    "шляпа", "пальто", "ботинок", "перчатка", "платье", "рубашка",
    "кровать", "подушка", "одеяло", "ковер", "занавеска",
]

# Filter to words that exist in vocabulary
word_to_idx = {w: i for i, w in enumerate(words)}
common_available = [(w, word_to_idx[w]) for w in common_russian if w in word_to_idx]
common_available = list(dict(common_available).items())  # deduplicate
print(f"Common words found in vocabulary: {len(common_available)} / {len(common_russian)}")

# Greedy search among common words
if len(common_available) >= 7:
    common_idxs = [idx for _, idx in common_available]
    n_common = len(common_idxs)
    common_mat = matrix[common_idxs]
    common_sims = common_mat @ common_mat.T

    # Find best pair
    best_d = -1
    best_p = None
    for i in range(n_common):
        for j in range(i+1, n_common):
            d = 1.0 - common_sims[i, j]
            if d > best_d:
                best_d = d
                best_p = [i, j]

    # Greedy
    selected_local = list(best_p)
    for step in range(2, 7):
        k = len(selected_local)
        current_pairs = k*(k-1)//2
        current_sum = sum(1.0 - common_sims[selected_local[i], selected_local[j]]
                          for i in range(k) for j in range(i+1,k))
        best_mean = -1
        best_c = -1
        npc = current_pairs + k
        for c in range(n_common):
            if c in selected_local:
                continue
            ns = current_sum + sum(1.0 - common_sims[c, s] for s in selected_local)
            m = ns / npc
            if m > best_mean:
                best_mean = m
                best_c = c
        selected_local.append(best_c)

    common_word_indices = [common_idxs[i] for i in selected_local]
    common_raw = compute_score(common_word_indices)
    common_adj = adjusted_score(common_raw)
    common_words_result = [words[i] for i in common_word_indices]
    print(f"Common words optimal set: {common_words_result}")
    print(f"  Raw: {common_raw:.2f}, Adjusted: {common_adj:.2f}")

    # Update results
    results["common_words_ceiling"] = {
        "words": common_words_result,
        "raw": round(common_raw, 4),
        "adjusted": round(common_adj, 4),
        "note": "From curated list of ~180 common Russian nouns/adjectives"
    }
    results["percentiles"]["common_pct_of_ceiling"] = round(
        common_raw / results["theoretical_ceiling"]["raw_score"] * 100, 1)

# ── Plot ──────────────────────────────────────────────────────────────────
ceiling_raw = results["theoretical_ceiling"]["raw_score"]
ceiling_adj = results["theoretical_ceiling"]["adjusted_score"]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Raw score distribution
ax = axes[0]
ax.hist(random_raw, bins=60, alpha=0.6, color="steelblue", edgecolor="white", label="Random 7-word sets (N=10,000)", density=True)
ax.axvline(87.9, color="orange", lw=2, ls="--", label=f"Mean human = 87.9")
ax.axvline(104.8, color="red", lw=2, ls="--", label=f"Best human = 104.8")
ax.axvline(ceiling_raw, color="darkgreen", lw=2.5, ls="-", label=f"Theoretical ceiling = {ceiling_raw:.1f}")
if "common_words_ceiling" in results:
    ax.axvline(results["common_words_ceiling"]["raw"], color="purple", lw=1.5, ls=":",
               label=f"Common-word ceiling = {results['common_words_ceiling']['raw']:.1f}")
ax.set_xlabel("Raw DAT Score (mean cosine distance x 100)", fontsize=12)
ax.set_ylabel("Density", fontsize=12)
ax.set_title("Raw DAT Score Distribution", fontsize=14)
ax.legend(fontsize=9, loc="upper left")
ax.set_xlim(55, 120)

# Adjusted score distribution
ax = axes[1]
ax.hist(random_adj, bins=60, alpha=0.6, color="steelblue", edgecolor="white", label="Random 7-word sets (N=10,000)", density=True)
ax.axvline(adjusted_score(87.9), color="orange", lw=2, ls="--", label=f"Mean human = {adjusted_score(87.9):.1f}")
ax.axvline(adjusted_score(104.8), color="red", lw=2, ls="--", label=f"Best human = {adjusted_score(104.8):.1f}")
ax.axvline(ceiling_adj, color="darkgreen", lw=2.5, ls="-", label=f"Theoretical ceiling = {ceiling_adj:.1f}")
ax.set_xlabel("Adjusted DAT Score", fontsize=12)
ax.set_ylabel("Density", fontsize=12)
ax.set_title("Adjusted DAT Score Distribution", fontsize=14)
ax.legend(fontsize=9, loc="upper left")
ax.set_xlim(50, 125)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "E1_score_distributions.png"), dpi=150, bbox_inches="tight")
print(f"\nPlot saved to {OUT_DIR}/E1_score_distributions.png")

# Save updated results
with open(os.path.join(OUT_DIR, "results_E1.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("Results updated.")

# ── Print final summary ──────────────────────────────────────────────────
print("\n" + "="*70)
print("FINAL SUMMARY — E1: Theoretical Maximum DAT Score")
print("="*70)
print(f"\nTheoretical ceiling:")
print(f"  Words: {results['theoretical_ceiling']['words']}")
print(f"  Raw: {results['theoretical_ceiling']['raw_score']}")
print(f"  Adjusted: {results['theoretical_ceiling']['adjusted_score']}")

print(f"\nCommon-word ceiling:")
print(f"  Words: {results['common_words_ceiling']['words']}")
print(f"  Raw: {results['common_words_ceiling']['raw']}")
print(f"  Adjusted: {results['common_words_ceiling']['adjusted']}")

print(f"\nRandom baseline (N=10,000):")
rb = results["random_baseline"]
print(f"  Raw — mean: {rb['raw_mean']}, std: {rb['raw_std']}, max: {rb['raw_max']}")
print(f"  Adj — mean: {rb['adj_mean']}, std: {rb['adj_std']}, max: {rb['adj_max']}")

print(f"\nPercentile of ceiling:")
p = results["percentiles"]
print(f"  Mean human (87.9): {p['mean_human_pct_of_ceiling']}% of ceiling")
print(f"  Best human (104.8): {p['best_human_pct_of_ceiling']}% of ceiling")
print(f"  Common-word ceiling: {p['common_pct_of_ceiling']}% of theoretical ceiling")

print(f"\nRandom baseline % of ceiling:")
print(f"  Mean random raw ({rb['raw_mean']}): {rb['raw_mean']/ceiling_raw*100:.1f}% of ceiling")
print(f"  Max random raw ({rb['raw_max']}): {rb['raw_max']/ceiling_raw*100:.1f}% of ceiling")
