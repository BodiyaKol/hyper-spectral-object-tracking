"""Run ViPT on doll3 and save a video with predicted bounding boxes."""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIPT_ROOT = PROJECT_ROOT / "third_party" / "ViPT"


def read_initial_box(path):
    values = path.read_text(encoding="utf-8").replace(",", " ").split()
    if len(values) != 4:
        raise ValueError(f"Expected x y width height in {path}")
    box = np.asarray([float(value) for value in values], dtype=np.float32)
    if not np.isfinite(box).all() or box[2] <= 0 or box[3] <= 0:
        raise ValueError(f"Invalid initial box in {path}: {values}")
    return box


def read_rgb(path):
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Could not read frame: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, default=PROJECT_ROOT / "data" / "doll3")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=VIPT_ROOT / "models" / "ViPT_deep_rgbt.pth",
        help="ViPT RGB-T checkpoint downloaded from the upstream model release",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "visualizations" / "doll3_vipt_cpu.mp4",
    )
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()

    sequence_dir = args.sequence.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not sequence_dir.is_dir():
        parser.error(f"Sequence directory does not exist: {sequence_dir}")
    if not checkpoint.is_file():
        parser.error(
            f"ViPT checkpoint not found: {checkpoint}. Download ViPT_deep_rgbt.pth from the upstream Models & Raw Results release."
        )
    if not VIPT_ROOT.is_dir():
        parser.error(f"ViPT checkout not found: {VIPT_ROOT}")

    frame_paths = sorted(sequence_dir.glob("*.jpg"))
    if not frame_paths:
        parser.error(f"No JPG frames found in {sequence_dir}")
    init_box = read_initial_box(sequence_dir / "init_rect.txt")

    first_frame = read_rgb(frame_paths[0])
    height, width = first_frame.shape[:2]
    x, y, box_width, box_height = init_box
    if x < 0 or y < 0 or x + box_width > width or y + box_height > height:
        parser.error(f"Initial box {init_box.tolist()} falls outside frame size {width}x{height}")

    sys.path.insert(0, str(VIPT_ROOT))
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

    torch.set_num_threads(max(1, args.threads))
    tracker = ViPTTrack(params)
    print(f"Using device: {tracker.device}")
    print("Smoke-test mode: RGB is duplicated into the second 3-channel input.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {args.output}")

    try:
        for frame_index, frame_path in enumerate(frame_paths):
            rgb = first_frame if frame_index == 0 else read_rgb(frame_path)
            if rgb.shape[:2] != (height, width):
                raise ValueError(f"Frame dimensions changed at {frame_path}")

            # Upstream ViPT expects two aligned RGB-like inputs, totaling 6 channels.
            model_input = np.concatenate((rgb, rgb), axis=2)
            if frame_index == 0:
                current_box = init_box.tolist()
                tracker.initialize(model_input, {"init_bbox": current_box})
                score = None
            else:
                prediction = tracker.track(model_input)
                current_box = prediction["target_bbox"]
                score = prediction.get("best_score")

            x, y, box_width, box_height = [round(value) for value in current_box]
            output_frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.rectangle(
                output_frame,
                (x, y),
                (x + box_width, y + box_height),
                (0, 0, 255),
                2,
            )
            label = f"frame {frame_index + 1}/{len(frame_paths)}"
            if score is not None:
                label += f" score={score:.3f}"
            cv2.putText(
                output_frame, label, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
            )
            writer.write(output_frame)

            if (frame_index + 1) % 50 == 0 or frame_index + 1 == len(frame_paths):
                print(f"Processed {frame_index + 1}/{len(frame_paths)} frames")
    finally:
        writer.release()

    print(f"Saved tracking video: {args.output}")


if __name__ == "__main__":
    main()
