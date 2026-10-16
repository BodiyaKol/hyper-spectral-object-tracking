# HOT2026 Data

The project uses the HOT2026 dataset. Download it from the official challenge source (Kaggle / [hsitracking.com](https://www.hsitracking.com/)) and follow the current competition terms. Dataset files are intentionally excluded from Git; do not redistribute them from this repository.

Expected local layout:

```text
data/
├── 2026training.csv   # training annotations from the challenge archive
├── automobile4/       # locally unpacked false-color JPG sequence for smoke tests
├── raw/HOT2026/       # other downloaded challenge data
├── interim/           # converted cubes and intermediate annotations
└── processed/         # model-ready sequences or cached features
```

To load the example sequence, pass `data/automobile4`, `data/2026training.csv`, and the annotation video ID `vis-automobile4` to `hsi_tracking.data.dataset.load_tracking_sequence`. The sequence loader checks that the number of JPG frames matches the number of CSV annotations. Local CSVs, archives, and the `automobile4/` directory are ignored by Git.

The proposal reports 405 training videos, 75 validation videos, and 75 test videos, recorded with VIS (16 bands), NIR (25 bands), and RedNIR (15 bands) sensors. Confirm the exact file layout and split metadata against the current official release before implementing the loader. Keep any access credentials outside the repository.