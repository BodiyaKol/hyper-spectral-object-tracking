"""Run the HSI-adapted ViPT tracker on one HSI sequence."""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hsi_tracking.data.dataset import load_hsi_sequence
from hsi_tracking.evaluation import (
    calculate_auc,
    calculate_center_distance,
    calculate_dp20,
    calculate_iou,
)
from hsi_tracking.models.spectral_projector import get_sensor_spec
from hsi_tracking.training import DEFAULT_ADAPTER_DIR
from hsi_tracking.models.tracker import (
    HSISequenceTracker,
    build_hsi_adapted_vipt,
    resolve_device,
)
from hsi_tracking.visualization import plot_tracking_metrics


def _display_limits(sequence, sensor, device, scale):
    spec = get_sensor_spec(sensor)
    indices = np.linspace(0, len(sequence) - 1, min(12, len(sequence)), dtype=int)
    samples = []
    for index in indices:
        cube = sequence[int(index)].hsi
        bands = [0, spec.expected_bands // 2, spec.expected_bands - 1]
        samples.append((cube[::8, ::8, bands] / scale).reshape(-1, 3))
    pixels = np.concatenate(samples, axis=0)
    lower = np.percentile(pixels, 1, axis=0)
    upper = np.percentile(pixels, 99, axis=0)
    if np.any(upper <= lower):
        upper = np.maximum(upper, lower + 1e-6)
    return lower, upper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--sensor", required=True, choices=("VIS", "NIR", "RedNIR"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validation-data", action="store_true")
    mode.add_argument("--test-data", action="store_true")
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "third_party" / "ViPT" / "models" / "ViPT_deep_rgbt.pth",
    )
    parser.add_argument("--adapter-checkpoint", type=Path)
    parser.add_argument(
        "--allow-untrained",
        action="store_true",
        help="run only a shape/smoke test without trained adapters",
    )
    parser.add_argument(
        "--false-color", type=Path, help="paired false-color frame folder for video visualization"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--input-scale", type=float, default=255.0)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()

    default_checkpoint = DEFAULT_ADAPTER_DIR / "adapter_latest.pth"
    if not args.adapter_checkpoint and not args.allow_untrained:
        if not default_checkpoint.is_file():
            parser.error(
                f"Pass --adapter-checkpoint, put weights in {default_checkpoint}, "
                "or use --allow-untrained for a smoke test"
            )
        args.adapter_checkpoint = default_checkpoint
    sequence = load_hsi_sequence(args.data, args.sensor, false_color_dir=args.false_color)
    if not len(sequence):
        parser.error(f"No HSI frames found in {args.data}")
    if args.test_data:
        if sequence.annotation_kind != "groundtruth" or len(sequence.boxes) != len(sequence):
            parser.error("--test-data requires one ground-truth BBox per frame")
        initial_box = sequence.boxes[0]
        ground_truth = sequence.boxes
        run_mode = "test"
    else:
        if sequence.annotation_kind != "init":
            parser.error("--validation-data requires init_rect.txt")
        initial_box = sequence.boxes[0]
        ground_truth = None
        run_mode = "validation"

    torch.set_num_threads(max(1, args.threads))
    device = resolve_device(args.device)
    model, cfg = build_hsi_adapted_vipt(
        args.sensor,
        args.base_checkpoint,
        device=device,
        input_scale=args.input_scale,
    )
    if args.adapter_checkpoint:
        model.load_adapters(args.adapter_checkpoint)
    else:
        print("WARNING: running untrained projector/adapter; this is only a smoke test")
    tracker = HSISequenceTracker(model, cfg, device=device)

    output_dir = args.output_dir or (
        PROJECT_ROOT / "results" / "full-model" / f"{Path(args.data).name}_{args.sensor}" / run_mode
    )
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    first = sequence[0]
    height, width = first.hsi.shape[:2]
    lower, upper = _display_limits(sequence, args.sensor, device, args.input_scale)
    video_path = output_dir / "tracking.mp4"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {video_path}")

    tracker.initialize(first.hsi, initial_box)
    predictions = []
    tracking_seconds = 0.0
    try:
        for index in range(len(sequence)):
            sample = first if index == 0 else sequence[index]
            if index == 0:
                prediction_box = list(initial_box)
            else:
                started = time.perf_counter()
                prediction = tracker.track(sample.hsi)
                tracking_seconds += time.perf_counter() - started
                prediction_box = prediction["target_bbox"]
            prediction_box = [float(value) for value in prediction_box]
            predictions.append(prediction_box)

            if sample.false_color is not None:
                frame = np.asarray(sample.false_color * 255, dtype=np.uint8)
            else:
                spec = get_sensor_spec(args.sensor)
                bands = [0, spec.expected_bands // 2, spec.expected_bands - 1]
                display = np.clip(
                    (sample.hsi[..., bands] / args.input_scale - lower) / (upper - lower), 0, 1
                )
                frame = np.asarray(display * 255, dtype=np.uint8)
            frame = np.ascontiguousarray(frame)
            if ground_truth is not None:
                gx, gy, gw, gh = [round(value) for value in ground_truth[index]]
                cv2.rectangle(frame, (gx, gy), (gx + gw, gy + gh), (0, 255, 0), 1)
            px, py, pw, ph = [round(value) for value in prediction_box]
            cv2.rectangle(frame, (px, py), (px + pw, py + ph), (0, 0, 255), 2)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if (index + 1) % 50 == 0 or index + 1 == len(sequence):
                print(f"Processed {index + 1}/{len(sequence)} frames")
    finally:
        writer.release()

    rows = []
    for index, (frame_path, box) in enumerate(zip(sequence.frame_paths, predictions)):
        row = {
            "frame": frame_path.name,
            "x": box[0],
            "y": box[1],
            "width": box[2],
            "height": box[3],
        }
        if ground_truth is not None:
            row["iou"] = calculate_iou(ground_truth[index], box)
            row["center_error"] = calculate_center_distance(ground_truth[index], box)
        rows.append(row)
    predictions_path = output_dir / "predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        csv_writer.writeheader()
        csv_writer.writerows(rows)

    metrics = {
        "sequence": Path(args.data).name,
        "sensor": args.sensor,
        "mode": run_mode,
        "frames": len(sequence),
        "input_channels": first.hsi.shape[-1],
        "device": str(device),
        "adapter_checkpoint": str(args.adapter_checkpoint) if args.adapter_checkpoint else None,
        "trained_adapters": args.adapter_checkpoint is not None,
        "fps": (len(sequence) - 1) / tracking_seconds if tracking_seconds > 0 else None,
    }
    if ground_truth is not None:
        metrics.update(
            {
                "AUC": calculate_auc(ground_truth, predictions),
                "DP@20": calculate_dp20(ground_truth, predictions),
                "mean_IoU": float(np.mean([row["iou"] for row in rows])),
                "mean_center_error_px": float(np.mean([row["center_error"] for row in rows])),
            }
        )
        plot_tracking_metrics(
            ground_truth, predictions, model_name="HSI-ViPT", output_dir=output_dir
        )
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Saved video:", video_path)
    print("Saved predictions:", predictions_path)
    print("Saved metrics:", metrics_path)


if __name__ == "__main__":
    main()
