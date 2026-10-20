"""Train the HSI spectral projector and CNN prompt adapter while ViPT stays frozen."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hsi_tracking.training import train_hsi_adapters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="HSI sequence folder or root of labeled HSI sequences",
    )
    parser.add_argument("--sensor", required=True, choices=("VIS", "NIR", "RedNIR"))
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "third_party" / "ViPT" / "models" / "ViPT_deep_rgbt.pth",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="stop after this many optimizer steps; 0 means full run",
    )
    parser.add_argument("--input-scale", type=float, default=255.0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()

    torch_root = Path(args.base_checkpoint).expanduser().resolve()
    if not torch_root.is_file():
        parser.error(f"Base ViPT checkpoint not found: {torch_root}")
    output_dir = args.output_dir or (
        PROJECT_ROOT / "results" / "full-model" / f"{args.data.name}_{args.sensor}" / "training"
    )

    import torch

    torch.set_num_threads(max(1, args.threads))
    history = train_hsi_adapters(
        data_root=args.data,
        sensor=args.sensor,
        base_checkpoint=torch_root,
        output_dir=output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        input_scale=args.input_scale,
        device=args.device,
    )
    print("Training complete. Final epoch:", history[-1])
    print("Adapter checkpoint:", Path(output_dir) / "adapter_latest.pth")
    print("Training history:", Path(output_dir) / "training_history.json")


if __name__ == "__main__":
    main()
