# DAT-RU: Russian Adaptation of the Divergent Association Task

**21,000 Attempts to Think Differently: A Large-Scale Russian Adaptation of the Divergent Association Task Reveals Practice Resistance and Lexical Predictors of Semantic Creativity**

## Overview

**Try the test:** [https://mool32.github.io/dat-ru/](https://mool32.github.io/dat-ru/)

DAT-RU is a Russian-language adaptation of the Divergent Association Task (DAT; [Olson et al., 2021](https://doi.org/10.1073/pnas.2022340118)), a rapid, automated measure of verbal creativity. Participants name 10 semantically unrelated Russian nouns; the score is the mean cosine distance between word embeddings (navec, 300d) of the first 7 valid words.

This repository contains:
- **Web application** — browser-based DAT-RU instrument (`app.js`, `index.html`, `data/`)
- **Analysis code** — all 16 analysis blocks (`analysis_A1/` through `analysis_F1/`)
- **Paper** — manuscript, figures, and LaTeX source (`paper/`)
- **Anonymized data** — 21,159 submissions (`paper/DAT_RU_anonymized_data.csv`)

## Key Findings

| Finding | Value |
|---------|-------|
| N submissions | 21,159 |
| Cronbach's alpha | .899 |
| Split-half (Spearman-Brown) | .696 [.689, .701] |
| Test-retest | .231 |
| Practice effect | None (i.i.d. model) |
| Strongest predictor | Category diversity (r = .47) |
| Theoretical ceiling | 110.5 (best human: 104.8) |

## Repository Structure

```
datcreativity/
├── app.js                  # Frontend application logic
├── index.html              # Web interface
├── data/                   # Runtime data (words.json, forms.json, matrix.bin)
├── scripts/                # Data preparation (prepare_data.py)
├── analysis_A1/            # Cross-linguistic calibration
├── analysis_A2/            # Raw vs adjusted score robustness
├── analysis_A3/            # User-agent collision analysis
├── analysis_A4/            # Split-half reliability
├── analysis_B1/            # Practice effects (mixed-effects)
├── analysis_B2/            # i.i.d. sampling model
├── analysis_B3/            # Incubation effects
├── analysis_C1/            # Word frequency analysis
├── analysis_C2/            # Semantic category diversity
├── analysis_C3/            # Positional / anchor effects
├── analysis_C4/            # Within-set structural topology
├── analysis_D3/            # Time-on-task
├── analysis_E1/            # Theoretical ceiling
├── analysis_E3/            # UMAP semantic map
├── analysis_E5/            # Optimal stopping
├── analysis_F1/            # Literature comparison
└── paper/                  # Manuscript, figures, data
    ├── DAT_RU_paper.md     # Full paper (Markdown)
    ├── DAT_RU_paper.tex    # LaTeX version
    ├── figures/             # Publication figures (PDF + PNG)
    └── DAT_RU_anonymized_data.csv
```

## Embedding Model

**navec** `navec_hudlit_v1_12B_500K_300d_100q` — 300-dimensional word vectors trained on 12 billion tokens of Russian literary texts (Natasha Project). Vocabulary filtered to 68,841 noun lemmas via pymorphy3. Vectors L2-normalized and quantized to int8 for browser delivery.

## Score Formula

```
raw = (100 / 21) × Σ (1 - cos(v_i, v_j))    for all 21 pairs of 7 words
adjusted = 47.4548 × (raw/100)^3.3820 + 48.5157
```

The power-law calibration maps Russian navec distances to the English GloVe baseline (~78).

## Citation

> [Authors]. (2026). 21,000 Attempts to Think Differently: A Large-Scale Russian Adaptation of the Divergent Association Task Reveals Practice Resistance and Lexical Predictors of Semantic Creativity. *[Journal]*.

## License

Code: MIT License. Data: CC BY 4.0. Embedding model (navec): see [natasha/navec](https://github.com/natasha/navec).
