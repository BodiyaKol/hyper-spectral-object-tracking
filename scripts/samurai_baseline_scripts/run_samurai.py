"""Run SAMURAI on false-color tracking sequences."""

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from hsi_tracking.evaluation import (
    calculate_auc,
    calculate_center_distance,
    calculate_dp20,
    calculate_iou,
)
from hsi_tracking.visualization import plot_tracking_metrics

CHECKPOINT = ROOT / "third_party/SAMURAI/sam2/checkpoints/sam2.1_hiera_tiny.pt"
CONFIG = "configs/samurai/sam2.1_hiera_t.yaml"


def load_bbox_file(path):
    """Read x, y, width, height coordinates."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing bounding box file: {path}")

    boxes = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        values = [float(v) for v in re.split(r"[\s,]+", line.strip())]
        if len(values) != 4:
            raise ValueError(f"Invalid BBox in {path}: {line}")
        boxes.append(values)

    if not boxes:
        raise ValueError(f"No bounding boxes found in {path}")

    return boxes


def get_frames(data_dir):
    frames = [p for p in data_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg"}]

    if not frames:
        raise ValueError(f"No JPG frames in {data_dir}")

    if not all(p.stem.isdigit() for p in frames):
        raise ValueError("Frame filenames must be numbered.")

    frames.sort(key=lambda p: int(p.stem))

    numbers = [int(p.stem) for p in frames]
    expected = list(range(numbers[0], numbers[0] + len(numbers)))
    if numbers != expected:
        raise ValueError("Frame numbering contains gaps or duplicates.")

    return frames


def mask_to_bbox(mask):
    """Convert binary segmentation mask into xywh bounding box."""
    ys, xs = np.where(mask)

    if len(xs) == 0:
        return [0.0, 0.0, 0.0, 0.0]

    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1

    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def save_video(frames, predictions, output_path, fps=25):
    first = cv2.imread(str(frames[0]))
    if first is None:
        raise ValueError("Cannot read first image.")

    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError("Cannot create output MP4 video.")

    try:
        for frame_path, box in zip(frames, predictions):
            image = cv2.imread(str(frame_path))
            if image is None:
                raise ValueError(f"Cannot read {frame_path}")

            x, y, w, h = (round(v) for v in box)

            if w > 0 and h > 0:
                cv2.rectangle(
                    image,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2,
                )

            writer.write(image)
    finally:
        writer.release()


def save_predictions(frames, predictions, output_path):
    with output_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["frame", "x", "y", "width", "height", "confidence"])

        for frame, box in zip(frames, predictions):
            # Upstream does not provide a calibrated BBox confidence.
            writer.writerow([frame.name, *box, ""])


def evaluate(gt_boxes, predictions, elapsed, output_dir):
    """Calculate metrics using the shared evaluation implementation."""
    ious = [calculate_iou(gt, pred) for gt, pred in zip(gt_boxes, predictions)]
    distances = [calculate_center_distance(gt, pred) for gt, pred in zip(gt_boxes, predictions)]

    metrics = {
        "AUC": calculate_auc(gt_boxes, predictions),
        "DP@20": calculate_dp20(gt_boxes, predictions),
        "mean_iou": float(np.mean(ious)),
        "mean_center_error": float(np.mean(distances)),
        "FPS": len(predictions) / elapsed,
        "total_frames": len(predictions),
    }

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    plot_tracking_metrics(
        ground_truth_boxes=gt_boxes,
        predicted_boxes=predictions,
        model_name="SAMURAI Tiny",
        output_dir=output_dir,
    )

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validation-data", action="store_true")
    mode.add_argument("--test-data", action="store_true")

    args = parser.parse_args()
    data_dir = args.data.resolve()

    if not data_dir.is_dir():
        parser.error(f"Sequence directory not found: {data_dir}")

    run_mode = "test" if args.test_data else "validation"
    frames = get_frames(data_dir)

    if args.test_data:
        gt_boxes = load_bbox_file(data_dir / "groundtruth_rect.txt")
        if len(gt_boxes) != len(frames):
            raise ValueError("GT count does not match frame count.")
        initial_bbox = gt_boxes[0]
    else:
        gt_boxes = None
        initial_bbox = load_bbox_file(data_dir / "init_rect.txt")[0]

    if initial_bbox[2] <= 0 or initial_bbox[3] <= 0:
        raise ValueError("Initial bounding box must have positive size.")

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {CHECKPOINT}")

    output_dir = ROOT / "results" / "runs" / f"{data_dir.name}_{run_mode}" / "samurai"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Start with CPU because upstream CUDA operations may not work on MPS.
    device = "cpu"
    print(f"Loading SAMURAI on {device}...")

    predictor = build_sam2_video_predictor(CONFIG, str(CHECKPOINT), device=device)

    if not predictor.samurai_mode:
        raise RuntimeError("SAMURAI mode is not enabled.")

    x, y, w, h = initial_bbox
    xyxy_box = [x, y, x + w, y + h]

    predictions = [None] * len(frames)

    start = time.perf_counter()

    with torch.inference_mode():
        state = predictor.init_state(
            video_path=str(data_dir),
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
            async_loading_frames=False,
        )

        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=0,
            obj_id=0,
            box=np.array(xyxy_box, dtype=np.float32),
        )

        # First frame is initialized from the supplied bounding box.
        predictions[0] = list(initial_bbox)

        for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
            if frame_idx == 0:
                continue

            if len(object_ids) != 1 or len(masks) != 1:
                raise RuntimeError("Expected one tracked object.")

            mask = masks[0, 0].detach().cpu().numpy() > 0
            predictions[frame_idx] = mask_to_bbox(mask)

            if (frame_idx + 1) % 25 == 0:
                print(f"Processed {frame_idx + 1}/{len(frames)} frames")

    elapsed = time.perf_counter() - start

    if any(box is None for box in predictions):
        raise RuntimeError("Missing predictions for some frames.")

    save_predictions(frames, predictions, output_dir / "predictions.csv")
    save_video(frames, predictions, output_dir / "tracking.mp4")

    print(f"Saved outputs to: {output_dir}")

    if args.test_data:
        metrics = evaluate(gt_boxes, predictions, elapsed, output_dir)
        print(json.dumps(metrics, indent=2))
    else:
        print("Validation mode: no GT metrics calculated.")


if __name__ == "__main__":
    main()
