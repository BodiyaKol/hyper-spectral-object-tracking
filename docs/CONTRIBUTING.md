# Contributing

## Development setup

Run `make setup` and activate `.venv`, then install a PyTorch build into that environment using the command recommended for your machine at [pytorch.org](https://pytorch.org/get-started/locally/).

## Changes

- Keep reusable code in `src/hsi_tracking/`; use `scripts/` for runnable project commands and notebooks for exploration.
- Put configurable paths and experiment settings in `config/default.yaml` rather than hardcoding them in modules.
- Never commit HOT2026 files, credentials, checkpoints, or generated prediction videos.
- Add or update focused tests for behavior changes. Run `make lint` and `make test` before opening a pull request.
- Record evaluation protocol, sensor, checkpoint, and metrics for every reported result.