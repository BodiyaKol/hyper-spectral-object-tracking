# Experiments

This file records measured local runs. HOT2026 data, checkpoints, and generated videos remain local and are not committed. The saved adapter history does not include training sequence names, so the two-sequence provenance below comes from experiment notes rather than the JSON artifact.

## HSI-VIS adapter training

- Sensor: VIS, 16 spectral channels.
- Training data: two complete HSI-VIS sequences; names were not retained in `training_history.json`.
- Valid adjacent frame pairs per epoch: 1,428; batch size: 1; 5 epochs; 1,428 optimizer updates per epoch.
- Trainable parameters: 538,056 of 93,896,217 total (0.573%); ViPT weights are frozen.
- Mean train loss by epoch: 0.21862, 0.15469, 0.13451, 0.12110, 0.11305.
- No validation loop, cross-validation, best-epoch selection, or held-out split manifest is included.

## Held-out HSI-VIS sequence: automobile4

`automobile4` (201 frames) was run with and without trained adapters on CPU. The experiment notes say this sequence was not among the two training sequences.

| Metric | Untrained adapter | Trained adapter |
|---|---:|---:|
| AUC | 0.03771 | 0.07025 |
| DP@20 | 0.03483 | 0.06965 |
| Mean IoU | 0.03354 | 0.06681 |
| Mean center error | 215.60 px | 155.61 px |
| Tracking speed | 3.29 FPS | 3.18 FPS |

Metrics improve, but absolute accuracy remains low and the tracker can drift onto a distractor. This is one sequence, not a benchmark-wide result.

## RGB/false-color ViPT smoke test

The `ball` run duplicates ordinary RGB into ViPT's two three-channel inputs; it is not HSI inference. Recorded metrics are AUC 0.68963 and DP@20 0.9552 over 625 frames. `doll3` is a 370-frame initialization-only validation run without full ground truth, so it has no accuracy metrics.