"""Render an HSI sequence through the learnable 1x1 spectral projector."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hsi_tracking.data.dataset import load_hsi_sequence
from hsi_tracking.data.x2cube import get_sensor_spec
from hsi_tracking.models.spectral_projector import SpectralProjector


def project_frame(projector, sequence, index, device):
    sample = sequence[index]
    cube = torch.from_numpy(sample.hsi).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        projected = projector(cube, sample.sensor)[0].cpu().numpy().transpose(1, 2, 0)
    return sample, projected


def display_limits(projector, sequence, device, sample_count=12):
    indices = np.linspace(0, len(sequence) - 1, min(sample_count, len(sequence)), dtype=int)
    sampled_pixels = []
    for index in indices:
        _, projected = project_frame(projector, sequence, int(index), device)
        sampled_pixels.append(projected[::8, ::8].reshape(-1, 3))
    pixels = np.concatenate(sampled_pixels, axis=0)
    lower = np.percentile(pixels, 1, axis=0)
    upper = np.percentile(pixels, 99, axis=0)
    if np.any(upper <= lower):
        raise ValueError(f"Cannot scale constant pseudo-RGB channels: p01={lower}, p99={upper}")
    return lower, upper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, required=True, help="Folder containing raw HSI mosaic frames"
    )
    parser.add_argument("--sensor", required=True, help="VIS, NIR, or RedNIR")
    parser.add_argument(
        "--false-color", type=Path, help="Optional paired false-color folder for comparison"
    )
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--output-root", type=Path, default=PROJECT_ROOT / "results" / "projector_testing"
    )
    args = parser.parse_args()

    sensor_spec = get_sensor_spec(args.sensor)
    sequence = load_hsi_sequence(args.data, sensor_spec.name, false_color_dir=args.false_color)
    if len(sequence) == 0:
        parser.error(f"No HSI frames found in {args.data}")

    first_sample = sequence[0]
    height, width, channels = first_sample.hsi.shape
    if channels != sensor_spec.expected_bands:
        parser.error(
            f"{sensor_spec.name} should have {sensor_spec.expected_bands} channels, got {channels}"
        )

    output_dir = (
        args.output_root.expanduser().resolve()
        / f"{Path(args.data).name}_{sensor_spec.name}_{channels}_channels"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(max(1, args.threads))
    device = torch.device("cpu")
    projector = SpectralProjector().to(device).eval()
    lower, upper = display_limits(projector, sequence, device)
    selected_bands = projector.projection_bands[sensor_spec.name]

    first_pseudo = np.clip((first_sample.hsi[..., selected_bands] - lower) / (upper - lower), 0, 1)
    preview = np.asarray(first_pseudo * 255, dtype=np.uint8)
    preview_path = output_dir / "first_pseudo_rgb.png"
    if not cv2.imwrite(str(preview_path), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Could not write preview: {preview_path}")

    video_path = output_dir / "pseudo_rgb.mp4"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video: {video_path}")

    raw_mins, raw_maxs = [], []
    pseudo_mins, pseudo_maxs = [], []
    try:
        for index in range(len(sequence)):
            sample, projected = project_frame(projector, sequence, index, device)
            if not np.isfinite(projected).all():
                raise ValueError(f"Non-finite pseudo-RGB values in frame {sample.frame_path}")
            raw_mins.append(float(sample.hsi.min()))
            raw_maxs.append(float(sample.hsi.max()))
            pseudo_mins.append(float(projected.min()))
            pseudo_maxs.append(float(projected.max()))

            display = np.clip((projected - lower) / (upper - lower), 0, 1)
            frame = np.ascontiguousarray(display * 255, dtype=np.uint8)
            if sample.bbox is not None:
                x, y, box_width, box_height = [round(value) for value in sample.bbox]
                cv2.rectangle(frame, (x, y), (x + box_width, y + box_height), (0, 255, 0), 1)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if (index + 1) % 50 == 0 or index + 1 == len(sequence):
                print(f"Processed {index + 1}/{len(sequence)} frames")
    finally:
        writer.release()

    metrics = {
        "sequence": Path(args.data).name,
        "sensor": sensor_spec.name,
        "input_channels": channels,
        "input_shape_hwc": [height, width, channels],
        "input_dtype": str(first_sample.hsi.dtype),
        "input_min": min(raw_mins),
        "input_max": max(raw_maxs),
        "pseudo_rgb_shape_hwc": [height, width, 3],
        "pseudo_rgb_dtype": "uint8 for video; float32 before display scaling",
        "pseudo_rgb_min": min(pseudo_mins),
        "pseudo_rgb_max": max(pseudo_maxs),
        "display_percentiles": {"low": lower.tolist(), "high": upper.tolist()},
        "initialization_bands_zero_based": list(selected_bands),
        "projector_parameters": sum(parameter.numel() for parameter in projector.parameters()),
        "projector_weights_trained": False,
        "frames": len(sequence),
        "fps": args.fps,
        "annotation_kind": sequence.annotation_kind,
        "false_color_paired": sequence.false_color_paths is not None,
        "preview": preview_path.name,
        "video": video_path.name,
        "note": "Untrained 1x1 convolution initialized to select low/middle/high bands; pseudo-color is not true RGB.",
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Sensor: {sensor_spec.name}, channels: {channels}")
    print(f"Projection bands (0-based): {selected_bands}")
    print(f"Saved preview: {preview_path}")
    print(f"Saved pseudo-color video: {video_path}")
    print(f"Saved diagnostics: {metrics_path}")


if __name__ == "__main__":
    main()
