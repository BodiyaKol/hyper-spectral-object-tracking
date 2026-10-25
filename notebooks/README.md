# Notebooks

Use numbered notebooks for exploratory work, keeping reusable pipeline code in `src/hsi_tracking/`.

Planned notebooks:

- `01_data_and_preprocessing.ipynb`: inspect HOT2026 data, annotations, sensor channels, and preprocessing.
- `02_hsi_adapter_training.ipynb`: check the runtime and ViPT checkpoint, select HSI sequences, run a bounded smoke test or full adapter training, inspect artifacts, and perform a short inference check.

Reusable pipeline logic belongs in `src/hsi_tracking/`; notebooks are experiment and demonstration workflows. The HSI notebook does not implement cross-validation or a validation loop during training.