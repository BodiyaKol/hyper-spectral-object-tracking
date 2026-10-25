# HOT2026 Data

HOT2026 is distributed by the challenge organizers. Obtain it from the official challenge source and follow its current access and redistribution terms. Raw data, archives, and locally prepared sequences are excluded from Git.

Expected local layout:

```text
data/
├── 2026training.csv   # training annotations from the challenge archive
├── automobile4/       # locally unpacked false-color JPG sequence for smoke tests
├── raw/HOT2026/       # other downloaded challenge data
├── interim/           # converted cubes and intermediate annotations
└── processed/         # model-ready sequences or cached features
```
## HSI sequence format

The HSI loader accepts sequence folders containing single-channel mosaic frames (`.png`, `.tif`, or `.tiff`) and one annotation file:

- `groundtruth_rect.txt`: one `x y width height` box per frame; required for training and labeled test evaluation.
- `init_rect.txt`: the initial box only; used for validation/inference when remaining ground truth is unavailable.

Example:

```text
data/
├── raw/HOT2026/training/HSI-VIS/
│   ├── automobile4/
│   │   ├── 0001.png
│   │   └── groundtruth_rect.txt
│   └── ...
├── automobile4/                 # optional false-color/JPG example
└── processed/                    # optional converted data/cache
```

Pass the root containing sequence folders to the trainer; it recursively finds HSI folders with `groundtruth_rect.txt`. The selected sensor must match the input: VIS has 16 bands, NIR 25, and RedNIR 15. False-color JPG frames are not spectral mosaics and must not be passed to the HSI loader as HSI input.

The local `automobile4` sample is useful for smoke and inference experiments, but metrics on training data are not held-out validation. Keep credentials outside the repository and do not commit organizer data.