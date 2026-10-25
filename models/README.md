# Model Checkpoints

Model files are local artifacts and are excluded from Git.

- `hsi_adapters/adapter_latest.pth`: most recently completed full HSI adapter training run.
- `hsi_adapters/training_history.json`: per-epoch loss and parameter counts for that run.

The upstream ViPT initialization checkpoint is stored separately at `third_party/ViPT/models/ViPT_deep_rgbt.pth`. It initializes the frozen RGB-T backbone/head; it is not an HSI-trained checkpoint. Review upstream license terms before redistributing it.