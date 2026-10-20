"""Run ViPT on one image sequence in validation or labeled test mode."""

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIPT_ROOT = PROJECT_ROOT / "third_party" / "ViPT"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hsi_tracking.evaluation import (
    calculate_auc,
    calculate_center_distance,
    calculate_dp20,
    calculate_iou,
    calculate_precision_curve,
    calculate_success_curve,
)


def _box_from_values(values, source):
    if len(values) != 4:
        raise ValueError(f"Expected x y width height in {source}")
    box = np.asarray([float(value) for value in values], dtype=np.float32)
    if not np.isfinite(box).all() or box[2] <= 0 or box[3] <= 0:
        raise ValueError(f"Invalid bounding box in {source}: {values}")
    return box


def read_initial_box(path):
    values = path.read_text(encoding="utf-8").replace(",", " ").split()
    return _box_from_values(values, path)


def read_ground_truth(path):
    boxes = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        values = re.split(r"[\s,]+", line.strip().rstrip(","))
        if not values or values == [""]:
            continue
        boxes.append(_box_from_values(values, f"{path}:{line_number}"))
    if not boxes:
        raise ValueError(f"No ground-truth boxes found in {path}")
    return boxes


def _frame_sort_key(path):
    if path.stem.isdigit():
        return 0, int(path.stem)
    return 1, path.name.casefold()


def read_rgb(path):
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Could not read frame: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _load_tracker(checkpoint, threads):
    if not VIPT_ROOT.is_dir():
        raise FileNotFoundError(f"ViPT checkout not found: {VIPT_ROOT}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"ViPT checkpoint not found: {checkpoint}")

    from lib.config.vipt.config import cfg, update_config_from_file
    from lib.test.tracker.vipt import ViPTTrack

    config_path = VIPT_ROOT / "experiments" / "vipt" / "deep_rgbt.yaml"
    update_config_from_file(str(config_path))
    params = SimpleNamespace(
        cfg=cfg,
        checkpoint=str(checkpoint),
        template_factor=cfg.TEST.TEMPLATE_FACTOR,
        template_size=cfg.TEST.TEMPLATE_SIZE,
        search_factor=cfg.TEST.SEARCH_FACTOR,
        search_size=cfg.TEST.SEARCH_SIZE,
        save_all_boxes=False,
        debug=0,
    )
    torch.set_num_threads(max(1, threads))
    return ViPTTrack(params)


def _save_plots(ground_truth_boxes, predicted_boxes, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    success_thresholds, success_rates = calculate_success_curve(ground_truth_boxes, predicted_boxes)
    precision_thresholds, precision_rates = calculate_precision_curve(
        ground_truth_boxes, predicted_boxes
    )

    plt.figure(figsize=(8, 5))
    plt.plot(success_thresholds, success_rates, linewidth=2)
    plt.xlabel("IoU threshold")
    plt.ylabel("Success rate")
    plt.title("ViPT success plot")
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    success_path = output_dir / "success_plot.png"
    plt.savefig(success_path, dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(precision_thresholds, precision_rates, linewidth=2)
    plt.axvline(x=20, linestyle="--", alpha=0.6, label="20 px")
    plt.xlabel("Center error (pixels)")
    plt.ylabel("Precision")
    plt.title("ViPT precision plot")
    plt.xlim(0, 50)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    precision_path = output_dir / "precision_plot.png"
    plt.savefig(precision_path, dpi=180)
    plt.close()
    return success_path, precision_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, required=True, help="Folder containing frames and sequence annotations"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--validation-data",
        action="store_true",
        help="Use init_rect.txt; save predictions without ground-truth metrics",
    )
    mode.add_argument(
        "--test-data",
        action="store_true",
        help="Initialize from groundtruth_rect.txt row 1; use later rows only for evaluation",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=VIPT_ROOT / "models" / "ViPT_deep_rgbt.pth",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()

    sequence_dir = args.data.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not sequence_dir.is_dir():
        parser.error(f"Sequence directory does not exist: {sequence_dir}")

    frame_paths = sorted(
        (
            path
            for path in sequence_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ),
        key=_frame_sort_key,
    )
    if not frame_paths:
        parser.error(f"No JPG/PNG frames found in {sequence_dir}")

    ground_truth_boxes = None
    if args.test_data:
        ground_truth_path = sequence_dir / "groundtruth_rect.txt"
        try:
            ground_truth_boxes = read_ground_truth(ground_truth_path)
        except (OSError, ValueError) as error:
            parser.error(str(error))
        if len(ground_truth_boxes) != len(frame_paths):
            parser.error(
                f"Frame/ground-truth count mismatch: {len(frame_paths)} frames, "
                f"{len(ground_truth_boxes)} boxes in {ground_truth_path}"
            )
        initial_box = ground_truth_boxes[0]
        run_mode = "test"
    else:
        init_path = sequence_dir / "init_rect.txt"
        try:
            initial_box = read_initial_box(init_path)
        except (OSError, ValueError) as error:
            parser.error(str(error))
        run_mode = "validation"

    first_frame = read_rgb(frame_paths[0])
    height, width = first_frame.shape[:2]
    x, y, box_width, box_height = initial_box
    if x < 0 or y < 0 or x + box_width > width or y + box_height > height:
        parser.error(
            f"Initial box {initial_box.tolist()} falls outside frame size {width}x{height}"
        )

    sys.path.insert(0, str(VIPT_ROOT))
    try:
        tracker = _load_tracker(checkpoint, args.threads)
    except (FileNotFoundError, ImportError, RuntimeError) as error:
        parser.error(str(error))

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else PROJECT_ROOT / "results" / "runs" / f"{sequence_dir.name}_{run_mode}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "tracking.mp4"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {video_path}")

    predicted_boxes = []
    confidences = []
    tracking_seconds = 0.0
    try:
        for frame_index, frame_path in enumerate(frame_paths):
            rgb = first_frame if frame_index == 0 else read_rgb(frame_path)
            if rgb.shape[:2] != (height, width):
                raise ValueError(f"Frame dimensions changed at {frame_path}")

            # Upstream ViPT expects two aligned RGB-like inputs, totaling six channels.
            model_input = np.concatenate((rgb, rgb), axis=2)
            if frame_index == 0:
                current_box = initial_box.tolist()
                tracker.initialize(model_input, {"init_bbox": current_box})
                confidence = float("nan")
            else:
                started = time.perf_counter()
                prediction = tracker.track(model_input)
                tracking_seconds += time.perf_counter() - started
                current_box = prediction["target_bbox"]
                confidence = float(prediction.get("best_score", float("nan")))

            current_box = [float(value) for value in current_box]
            predicted_boxes.append(current_box)
            confidences.append(confidence)

            output_frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if ground_truth_boxes is not None:
                gx, gy, gw, gh = [round(value) for value in ground_truth_boxes[frame_index]]
                cv2.rectangle(output_frame, (gx, gy), (gx + gw, gy + gh), (0, 255, 0), 1)
            px, py, pw, ph = [round(value) for value in current_box]
            cv2.rectangle(output_frame, (px, py), (px + pw, py + ph), (0, 0, 255), 2)
            label = f"{run_mode} frame {frame_index + 1}/{len(frame_paths)}"
            if np.isfinite(confidence):
                label += f" score={confidence:.3f}"
            cv2.putText(
                output_frame, label, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
            )
            writer.write(output_frame)

            if (frame_index + 1) % 50 == 0 or frame_index + 1 == len(frame_paths):
                print(f"Processed {frame_index + 1}/{len(frame_paths)} frames")
    finally:
        writer.release()

    result_rows = []
    for index, (frame_path, predicted_box, confidence) in enumerate(
        zip(frame_paths, predicted_boxes, confidences)
    ):
        row = {
            "frame": frame_path.name,
            "pred_x": predicted_box[0],
            "pred_y": predicted_box[1],
            "pred_width": predicted_box[2],
            "pred_height": predicted_box[3],
            "confidence": confidence,
        }
        if ground_truth_boxes is not None:
            gt_box = ground_truth_boxes[index]
            row.update(
                {
                    "gt_x": gt_box[0],
                    "gt_y": gt_box[1],
                    "gt_width": gt_box[2],
                    "gt_height": gt_box[3],
                    "iou": calculate_iou(gt_box, predicted_box),
                    "center_error": calculate_center_distance(gt_box, predicted_box),
                }
            )
        result_rows.append(row)

    csv_path = output_dir / "predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as output_file:
        writer_csv = csv.DictWriter(output_file, fieldnames=list(result_rows[0]))
        writer_csv.writeheader()
        writer_csv.writerows(result_rows)

    metrics = {
        "mode": run_mode,
        "sequence": sequence_dir.name,
        "frames": len(frame_paths),
        "device": str(tracker.device),
        "input_note": "RGB is duplicated to form the two 3-channel ViPT inputs; this is a smoke test, not HSI inference.",
        "tracking_fps": (len(frame_paths) - 1) / tracking_seconds if tracking_seconds > 0 else None,
    }
    description_path = sequence_dir / "description.txt"
    if description_path.is_file():
        metrics["sequence_attributes"] = description_path.read_text(encoding="utf-8").splitlines()

    if ground_truth_boxes is not None:
        ious = [calculate_iou(gt, pred) for gt, pred in zip(ground_truth_boxes, predicted_boxes)]
        center_errors = [
            calculate_center_distance(gt, pred)
            for gt, pred in zip(ground_truth_boxes, predicted_boxes)
        ]
        metrics.update(
            {
                "AUC": calculate_auc(ground_truth_boxes, predicted_boxes),
                "DP@20": calculate_dp20(ground_truth_boxes, predicted_boxes),
                "mean_IoU": float(np.mean(ious)),
                "mean_center_error_px": float(np.mean(center_errors)),
            }
        )
        _save_plots(ground_truth_boxes, predicted_boxes, output_dir)

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Using device: {tracker.device}")
    print(f"Saved video: {video_path}")
    print(f"Saved predictions: {csv_path}")
    print(f"Saved summary: {metrics_path}")
    if ground_truth_boxes is not None:
        fps_text = f"{metrics['tracking_fps']:.2f}" if metrics["tracking_fps"] else "n/a"
        print(
            f"AUC={metrics['AUC']:.4f} DP@20={metrics['DP@20']:.4f} "
            f"mean_IoU={metrics['mean_IoU']:.4f} "
            f"mean_center_error={metrics['mean_center_error_px']:.2f}px FPS={fps_text}"
        )


if __name__ == "__main__":
    main()
