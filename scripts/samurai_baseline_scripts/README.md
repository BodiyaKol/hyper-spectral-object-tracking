# SAMURAI False-Color Baseline

Standalone SAMURAI baseline for hyperspectral object tracking using false-color RGB frames.

## Installation

Run from the repository root:

```bash
bash scripts/samurai_baseline_scripts/install_samurai.sh
source .venv-samurai/bin/activate
```

## Checkpoint

Download the SAM 2.1 Hiera Tiny checkpoint:

https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt

Save it to:

`third_party/SAMURAI/sam2/checkpoints/sam2.1_hiera_tiny.pt`

Configuration: `configs/samurai/sam2.1_hiera_t.yaml`

## Run test sequence

```bash
python scripts/samurai_baseline_scripts/run_samurai.py \
  --data data/samurai/ball \
  --test-data
```

Expected outputs in `results/runs/ball_test/samurai/`:

- `tracking.mp4`
- `predictions.csv`
- `metrics.json`
- `success_plot.png`
- `precision_plot.png`

## Run validation sequence

```bash
python scripts/samurai_baseline_scripts/run_samurai.py \
  --data data/samurai/doll3 \
  --validation-data
```

Validation outputs contain `tracking.mp4` and `predictions.csv`. Metrics are not calculated without ground truth.

## Environment

Locally tested on macOS ARM64, Python 3.12.5, PyTorch 2.14.1, CPU inference.

SAMURAI mode is enabled. The optional SAM 2 CUDA extension is not available on macOS, so some post-processing functionality may be unavailable.

## Local test results

**ball (625 frames)**

- AUC: 0.7608
- DP@20: 0.9792
- Mean IoU: 0.7605
- Mean center error: 6.84 pixels
- FPS: 0.566

**doll3 (370 frames)**

- Validation completed successfully.
- No ground-truth metrics calculated.

These are local baseline results and have not yet been verified against the official HOT2026 evaluator.

Checkpoints, upstream SAMURAI code, datasets and generated outputs are excluded from Git.