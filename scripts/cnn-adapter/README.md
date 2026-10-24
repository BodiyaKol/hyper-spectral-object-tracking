# CNN Adapter Experiments

These scripts patch the ignored upstream ViPT checkout for CPU + external HSI prompts, train the spectral projector/CNN prompt adapter, and run HSI inference.

## One-time setup

Run from the project root with the dedicated Python 3.7 `vipt` Conda environment active:

```bash
python scripts/cnn-adapter/patch_vipt_hsi.py
```

The patcher first applies CPU compatibility fixes, then adds `external_prompts` to the ViPT forward path. In HSI mode the projected 3-channel template/search go through the RGB patch embedding; a prompt tensor is added before every transformer block. Candidate elimination is disabled for this mode to preserve prompt/token alignment. The patcher is idempotent. The base checkpoint is expected at `third_party/ViPT/models/ViPT_deep_rgbt.pth`.

## CPU and Colab GPU

Both training and inference accept `--device auto|cpu|cuda`; `auto` selects CUDA when `torch.cuda.is_available()` is true, otherwise CPU. The local `requirements-vipt-cpu.txt` installs CPU-only Torch and must **not** be used in a GPU Colab runtime.

In Colab, first select **Runtime → Change runtime type → GPU**, clone this project and ViPT, mount Drive, then install the other dependencies without replacing Colab's Torch:

```bash
!pip install -r requirements-vipt-colab.txt
!python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
!python scripts/cnn-adapter/patch_vipt_hsi.py
```

The GPU check must print `True` before training. Keep the dataset on mounted Drive or copy a working subset into `/content`; set `--data` to a folder containing sequence subfolders with HSI mosaics and `groundtruth_rect.txt`. Put the ViPT checkpoint at `third_party/ViPT/models/ViPT_deep_rgbt.pth`.

Example Colab smoke training (change `--data` to the actual Drive folder):

```bash
!python scripts/cnn-adapter/train_hsi_adapters.py \
  --data /content/drive/MyDrive/HOT2026/train/VIS \
  --sensor VIS --epochs 1 --max-steps 10 --device auto \
  --output-dir /content/drive/MyDrive/hsi-results/automobile4_VIS/training_smoke
```

The training CLI and model automatically move tensors and weights to the selected device. CUDA execution is supported by the current code path, but has not yet been validated on an actual Colab GPU; the upstream ViPT code and its older `timm` dependency may need compatibility fixes in Colab's current runtime. The 10-step example verifies execution only, not tracking quality.

## Training

Each sequence folder must contain HSI mosaic frames and `groundtruth_rect.txt` with one `x y width height` box per frame. Start with a one-step pipeline check:

```bash
python scripts/cnn-adapter/train_hsi_adapters.py \
  --data data/automobile4 \
  --sensor VIS \
  --epochs 1 \
  --max-steps 1 \
  --output-dir results/full-model/automobile4_VIS/training_smoke
```

This performs only one optimizer step; it does **not** train a useful tracker. For an actual experiment, remove `--max-steps 1`, choose a suitable `--epochs` value, and train on the training split. Adapter weights and `training_history.json` are saved to the output directory. Defaults use CPU, batch size 1, learning rate `1e-4`, and raw HSI input divided by `255`; confirm the sensor's data scale before a full run. CPU is intended for smoke tests; full training will be very slow.

Training pairs adjacent labeled frames. The first frame is the template; the next frame is the search crop. The tracker backbone and prediction head stay frozen; only projector and prompt-adapter parameters are optimized with Smooth L1 plus GIoU loss. The adapter emits 12 prompt maps of `B×N×768` for ViT-Base, one before each transformer block.

## Inference

Run with the saved adapter checkpoint. For labeled test data, the first GT box initializes tracking; later boxes are only used for metrics:

```bash
python scripts/cnn-adapter/run_hsi_vipt.py \
  --data data/automobile4 \
  --sensor VIS \
  --test-data \
  --adapter-checkpoint results/full-model/automobile4_VIS/training_smoke/adapter_latest.pth \
  --output-dir results/full-model/automobile4_VIS/test_smoke
```

By default, training saves to `results/full-model/<sequence>_<sensor>/training/`; inference saves to `results/full-model/<sequence>_<sensor>/<mode>/`. Outputs include `tracking.mp4`, `predictions.csv`, `metrics.json`, and test-mode plots. `--allow-untrained` is only for shape/forward smoke tests and produces arbitrary predictions. Generated full-model artifacts are ignored by Git.

## Current limitation

Upstream ViPT is pretrained for RGB plus a second 3-channel modality. The HSI path is an experimental adaptation: the cube is projected to pseudo-RGB and CNN-generated prompts are injected into the frozen transformer. There is no official upstream HSI checkpoint. Until the adapters are trained and validated, predictions are not meaningful HSI results. CPU training is only suitable for tiny smoke tests.