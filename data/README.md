# HOT2026 Data

The project uses the HOT2026 dataset. Download it from the official challenge source (Kaggle / [hsitracking.com](https://www.hsitracking.com/)) and follow the current competition terms. Dataset files are intentionally excluded from Git; do not redistribute them from this repository.

Expected local layout:

```text
data/
├── raw/HOT2026/       # downloaded challenge data
├── interim/           # converted cubes and intermediate annotations
└── processed/         # model-ready sequences or cached features
```

The proposal reports 405 training videos, 75 validation videos, and 75 test videos, recorded with VIS (16 bands), NIR (25 bands), and RedNIR (15 bands) sensors. Confirm the exact file layout and split metadata against the current official release before implementing the loader. Keep any access credentials outside the repository.