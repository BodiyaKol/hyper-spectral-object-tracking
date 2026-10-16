"""Save the first frame with its initialization bounding box."""

import argparse
from pathlib import Path

from hsi_tracking.visualization import save_initial_frame_preview


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, default=Path("data/doll3"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    output_path = save_initial_frame_preview(args.sequence, args.output)
    print(f"Saved preview: {output_path}")


if __name__ == "__main__":
    main()
