# HSI ViPT Adapter

These scripts adapt upstream ViPT for hyperspectral mosaics. A sensor-specific 1×1 spectral projector makes three pseudo-RGB channels; a CNN generates external prompts for each ViT block. The ViPT backbone and bbox head are initialized from `ViPT_deep_rgbt.pth` and frozen; training updates only the projector and prompt adapter.

The HSI path is experimental, not an official upstream HSI model. It has completed a five-epoch Colab T4 training run, but tracking quality remains low on the one documented held-out sequence. There is no validation loop or cross-validation during training.

## Environment and patching

Run commands from the repository root. The local `requirements-vipt-cpu.txt` describes the legacy Python 3.7/PyTorch 1.7 CPU environment. For Colab GPU, select a GPU runtime and install `requirements-vipt-colab.txt`, which deliberately leaves Colab's CUDA PyTorch intact. Do not install the CPU requirements or run upstream `install_vipt.sh` in a GPU runtime.

Clone upstream ViPT and prepare the checkout once per environment:

```bash
git clone https://github.com/jiawen-zhu/ViPT.git third_party/ViPT
python scripts/cnn-adapter/patch_vipt_hsi.py
```

The patcher applies CPU/device compatibility changes and adds `external_prompts` to the ViPT transformer path. It disables candidate elimination in HSI prompt mode to keep prompt/token alignment. The patcher is idempotent. Put the RGB-T initialization checkpoint at `third_party/ViPT/models/ViPT_deep_rgbt.pth`; it initializes ViPT but is not an HSI-trained model.

In Colab, confirm CUDA before training:

```python
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
```

The recorded 5-epoch run completed on a Colab T4 with Python 3.13/PyTorch 2.11 CUDA. Other upstream/PyTorch combinations may expose compatibility differences.

## Data and training

Each HSI sequence directory needs mosaic frames (`.png`, `.tif`, or `.tiff`) and `groundtruth_rect.txt` with one `x y width height` bbox per frame. Pass either a sequence directory or a root containing sequence directories to `--data`; discovery is recursive.

```bash
python scripts/cnn-adapter/train_hsi_adapters.py \
  --data /path/to/HOT2026/training/HSI-VIS \
  --sensor VIS \
  --base-checkpoint third_party/ViPT/models/ViPT_deep_rgbt.pth \
  --epochs 5 --device auto \
  --output-dir results/full-model/hsi_vis/training
```

Use `--max-steps 1` for a pipeline smoke test, not a quality run. Defaults are batch size 1, learning rate `1e-4`, input scale 255, and zero DataLoader workers. Adjacent valid GT frames form template/search pairs; the DataLoader shuffles these pairs each epoch. Loss is Smooth L1 plus `2 × (1 - GIoU)`, with gradient clipping at 0.1.

The trainer records `adapter_latest.pth` and `training_history.json` in `--output-dir`. A full run (`--max-steps 0`) also overwrites the local latest copy in `models/hsi_adapters/`; smoke runs do not. It starts from the base ViPT checkpoint each run; optimizer state is not resumed. In Colab, copy outputs to persistent Drive before the runtime ends.

## Inference and outputs

Labeled test sequence:

```bash
python scripts/cnn-adapter/run_hsi_vipt.py \
  --data /path/to/labeled_sequence --sensor VIS --test-data \
  --adapter-checkpoint models/hsi_adapters/adapter_latest.pth \
  --output-dir results/full-model/example_VIS/test --device auto
```

Validation sequence with only `init_rect.txt`:

```bash
python scripts/cnn-adapter/run_hsi_vipt.py \
  --data /path/to/validation_sequence --sensor VIS --validation-data \
  --adapter-checkpoint models/hsi_adapters/adapter_latest.pth \
  --output-dir results/full-model/example_VIS/validation --device auto
```

If `--adapter-checkpoint` is omitted, inference uses `models/hsi_adapters/adapter_latest.pth`. Test mode produces `tracking.mp4`, `predictions.csv`, `metrics.json`, `success_plot.png`, and `precision_plot.png`; validation mode cannot compute accuracy without full ground truth. `--allow-untrained` is for forward/shape smoke tests only.

## Known limitations

- Training loss is measured on training pairs; no validation metrics are computed during epochs.
- Training search crops are centered using the previous-frame ground-truth bbox, so training does not fully simulate tracker drift.
- The saved 10-video run improved AUC from 0.0377 to 0.0702 on `automobile4`, but the absolute score remains low and the tracker can switch to a distractor.
- CPU is practical for inference/smoke checks, but full training is slow.