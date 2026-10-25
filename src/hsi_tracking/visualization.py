"""Utilities for rendering tracking predictions and initialization previews."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

import matplotlib.pyplot as plt

from hsi_tracking.evaluation import (
    calculate_auc,
    calculate_dp20,
    calculate_success_curve,
    calculate_precision_curve,
)


def save_initial_frame_preview(
    sequence_dir: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Draw ``init_rect.txt`` on the first image and save the preview as PNG."""
    sequence_dir = Path(sequence_dir)
    frame_paths = sorted(
        (
            path
            for path in sequence_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ),
        key=lambda path: (
            not path.stem.isdigit(),
            int(path.stem) if path.stem.isdigit() else path.name,
        ),
    )
    if not frame_paths:
        raise FileNotFoundError(f"No image frames found in {sequence_dir}")

    init_rect_path = sequence_dir / "init_rect.txt"
    values = init_rect_path.read_text(encoding="utf-8").replace(",", " ").split()
    if len(values) != 4:
        raise ValueError(f"Expected x y width height in {init_rect_path}")
    x, y, width, height = map(float, values)
    if width <= 0 or height <= 0:
        raise ValueError(f"Bounding-box width and height must be positive: {values}")

    with Image.open(frame_paths[0]) as source:
        image = source.convert("RGB")
    right, bottom = x + width, y + height
    if x < 0 or y < 0 or right > image.width or bottom > image.height:
        raise ValueError(f"Bounding box {values} falls outside the first frame {image.size}")

    draw = ImageDraw.Draw(image)
    draw.rectangle((round(x), round(y), round(right), round(bottom)), outline="red", width=3)

    output_path = Path(output_path or f"results/visualizations/{sequence_dir.name}_init.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


def plot_tracking_metrics(
    ground_truth_boxes,
    predicted_boxes,
    model_name="SAMURAI",
    output_dir="results/plots",
):
    """Generate Success and Precision plots for tracking evaluation."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate evaluation metrics
    auc = calculate_auc(ground_truth_boxes, predicted_boxes)
    dp20 = calculate_dp20(ground_truth_boxes, predicted_boxes)

    # Calculate plot data
    success_thresholds, success_rates = calculate_success_curve(
        ground_truth_boxes, predicted_boxes
    )

    precision_thresholds, precision_rates = calculate_precision_curve(
        ground_truth_boxes, predicted_boxes
    )

    # Success Plot
    plt.figure(figsize=(8, 5))
    plt.plot(
        success_thresholds,
        success_rates,
        linewidth=2,
        label=f"{model_name} (AUC={auc:.3f})",
    )

    plt.xlabel("IoU Threshold")
    plt.ylabel("Success Rate")
    plt.title("Tracking Success Plot")
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    success_path = output_dir / "success_plot.png"
    plt.savefig(success_path, dpi=300)
    plt.close()

    # Precision Plot
    plt.figure(figsize=(8, 5))
    plt.plot(
        precision_thresholds,
        precision_rates,
        linewidth=2,
        label=f"{model_name} (DP@20={dp20:.3f})",
    )

    plt.axvline(
        x=20,
        linestyle="--",
        alpha=0.6,
        label="20 px threshold",
    )

    plt.xlabel("Distance Threshold (pixels)")
    plt.ylabel("Precision")
    plt.title("Tracking Precision Plot")
    plt.xlim(0, 50)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    precision_path = output_dir / "precision_plot.png"
    plt.savefig(precision_path, dpi=300)
    plt.close()

    return success_path, precision_path
