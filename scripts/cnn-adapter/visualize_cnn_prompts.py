"""Visualize pseudo-RGB input and untrained CNN prompt activations for HSI."""

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
from hsi_tracking.models.prompt_adapter import CNNPromptAdapter
from hsi_tracking.models.spectral_projector import SpectralProjector


def project_cube(projector, cube_hwc, sensor):
    cube = torch.from_numpy(cube_hwc.transpose(2, 0, 1).copy()).unsqueeze(0)
    with torch.no_grad():
        return projector(cube, sensor)[0].cpu().numpy().transpose(1, 2, 0)


def display_limits(projector, sequence, sensor, sample_count=12):
    indices = np.linspace(0, len(sequence) - 1, min(sample_count, len(sequence)), dtype=int)
    sampled = []
    for index in indices:
        sample = sequence[int(index)]
        projected = project_cube(projector, sample.hsi, sensor)
        sampled.append(projected[::8, ::8].reshape(-1, 3))
    pixels = np.concatenate(sampled, axis=0)
    lower = np.percentile(pixels, 1, axis=0)
    upper = np.percentile(pixels, 99, axis=0)
    if np.any(upper <= lower):
        raise ValueError(f"Cannot scale constant pseudo-RGB channels: p01={lower}, p99={upper}")
    return lower, upper


def to_display_rgb(image, lower, upper):
    scaled = np.clip((image - lower) / (upper - lower), 0, 1)
    return np.ascontiguousarray(scaled * 255, dtype=np.uint8)


def prompt_activation(prompt, grid_size):
    height, width = grid_size
    tokens = prompt[0].detach().cpu().numpy().reshape(height, width, -1)
    return np.mean(np.abs(tokens), axis=-1)


def colorize_activation(activation, upper_limit, output_size):
    normalized = np.clip(activation / max(upper_limit, 1e-8), 0, 1)
    small = np.asarray(normalized * 255, dtype=np.uint8)
    resized = cv2.resize(small, output_size, interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(cv2.applyColorMap(resized, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)


def save_prompt_grid(maps, output_path):
    tile_width, tile_height = 320, 220
    canvas = np.zeros((tile_height * 3, tile_width * 4, 3), dtype=np.uint8)
    for index, activation in enumerate(maps):
        upper = max(float(np.percentile(activation, 99)), 1e-8)
        heatmap = colorize_activation(activation, upper, (tile_width, tile_height - 28))
        row, col = divmod(index, 4)
        y, x = row * tile_height, col * tile_width
        canvas[y : y + tile_height - 28, x : x + tile_width] = heatmap
        cv2.putText(
            canvas,
            f"Prompt layer {index}",
            (x + 8, y + tile_height - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    if not cv2.imwrite(str(output_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Could not write prompt grid: {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, required=True, help="Folder containing HSI mosaic frames"
    )
    parser.add_argument("--sensor", required=True, help="VIS, NIR or RedNIR")
    parser.add_argument("--layer", type=int, default=0, help="Prompt layer to overlay (0-based)")
    parser.add_argument("--stride", type=int, default=16, help="HSI-to-ViPT prompt-grid stride")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "cnn_adapter_testing",
    )
    args = parser.parse_args()

    spec = get_sensor_spec(args.sensor)
    sequence = load_hsi_sequence(args.data, spec.name)
    if not len(sequence):
        parser.error(f"No HSI frames found in {args.data}")
    if args.layer < 0 or args.layer >= 12:
        parser.error("--layer must be between 0 and 11 for ViPT base")
    if args.stride < 1:
        parser.error("--stride must be positive")

    first = sequence[0]
    height, width, channels = first.hsi.shape
    if channels != spec.expected_bands:
        parser.error(f"{spec.name} expects {spec.expected_bands} channels, got {channels}")

    grid_size = (max(1, height // args.stride), max(1, width // args.stride))
    output_dir = (
        args.output_root.expanduser().resolve()
        / f"{Path(args.data).name}_{spec.name}_{channels}_channels"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(max(1, args.threads))
    projector = SpectralProjector().eval()
    adapter = CNNPromptAdapter(embed_dim=768, num_layers=12).eval()
    selected_bands = projector.projection_bands[spec.name]
    lower, upper = display_limits(projector, sequence, spec.name)

    first_projected = project_cube(projector, first.hsi, spec.name)
    first_pseudo_rgb = to_display_rgb(first_projected, lower, upper)
    first_tensor = torch.from_numpy(first.hsi.transpose(2, 0, 1).copy()).unsqueeze(0)
    with torch.no_grad():
        first_prompts = adapter(first_tensor, spec.name, grid_size)
    first_maps = [prompt_activation(prompt, grid_size) for prompt in first_prompts]
    selected_map = first_maps[args.layer]
    activation_limit = max(float(np.percentile(selected_map, 99)), 1e-8)

    pseudo_path = output_dir / "pseudo_rgb_before_adapter.png"
    heatmap_path = output_dir / f"prompt_layer_{args.layer:02d}.png"
    overlay_path = output_dir / f"prompt_overlay_layer_{args.layer:02d}.png"
    grid_path = output_dir / "all_prompt_layers.png"
    for path, image in ((pseudo_path, first_pseudo_rgb),):
        if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"Could not write image: {path}")

    selected_heatmap = colorize_activation(selected_map, activation_limit, (width, height))
    overlay = cv2.addWeighted(first_pseudo_rgb, 0.55, selected_heatmap, 0.45, 0)
    if first.bbox is not None:
        x, y, box_width, box_height = [round(value) for value in first.bbox]
        cv2.rectangle(overlay, (x, y), (x + box_width, y + box_height), (0, 255, 0), 2)
    if not cv2.imwrite(str(heatmap_path), cv2.cvtColor(selected_heatmap, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Could not write prompt heatmap: {heatmap_path}")
    if not cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Could not write prompt overlay: {overlay_path}")
    save_prompt_grid(first_maps, grid_path)

    video_path = output_dir / "cnn_prompt_diagnostic.mp4"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width * 2, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video: {video_path}")

    activation_means = []
    try:
        for index in range(len(sequence)):
            sample = first if index == 0 else sequence[index]
            projected = project_cube(projector, sample.hsi, spec.name)
            pseudo_rgb = to_display_rgb(projected, lower, upper)
            cube = torch.from_numpy(sample.hsi.transpose(2, 0, 1).copy()).unsqueeze(0)
            with torch.no_grad():
                prompts = adapter(cube, spec.name, grid_size)
            activation = prompt_activation(prompts[args.layer], grid_size)
            activation_means.append(float(activation.mean()))
            heatmap = colorize_activation(activation, activation_limit, (width, height))
            prompt_overlay = cv2.addWeighted(pseudo_rgb, 0.55, heatmap, 0.45, 0)
            if sample.bbox is not None:
                x, y, box_width, box_height = [round(value) for value in sample.bbox]
                cv2.rectangle(
                    prompt_overlay, (x, y), (x + box_width, y + box_height), (0, 255, 0), 2
                )
            side_by_side = np.concatenate((pseudo_rgb, prompt_overlay), axis=1)
            writer.write(cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR))
            if (index + 1) % 50 == 0 or index + 1 == len(sequence):
                print(f"Processed {index + 1}/{len(sequence)} frames")
    finally:
        writer.release()

    metrics = {
        "sequence": Path(args.data).name,
        "sensor": spec.name,
        "input_channels": channels,
        "input_shape_hwc": [height, width, channels],
        "input_dtype": str(first.hsi.dtype),
        "input_min": float(min(sample.hsi.min() for sample in (first, sequence[-1]))),
        "input_max": float(max(sample.hsi.max() for sample in (first, sequence[-1]))),
        "prompt_layers": len(first_prompts),
        "prompt_embed_dim": adapter.embed_dim,
        "prompt_grid": list(grid_size),
        "prompt_tokens_per_layer": grid_size[0] * grid_size[1],
        "adapter_parameters": sum(parameter.numel() for parameter in adapter.parameters()),
        "adapter_weights_trained": False,
        "displayed_layer": args.layer,
        "activation_mean_first_frame": float(first_maps[args.layer].mean()),
        "activation_mean_last_frame": activation_means[-1],
        "initialization_bands_zero_based": list(selected_bands),
        "frames": len(sequence),
        "fps": args.fps,
        "note": "Untrained prompts show random-initialization activations, not learned object attention.",
        "outputs": [
            pseudo_path.name,
            heatmap_path.name,
            overlay_path.name,
            grid_path.name,
            video_path.name,
        ],
    }
    metrics_path = output_dir / "diagnostics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Sensor: {spec.name}, input channels: {channels}")
    print(f"Prompt shape per layer: (1, {grid_size[0] * grid_size[1]}, {adapter.embed_dim})")
    print(f"Adapter parameters: {metrics['adapter_parameters']:,} (trainable, untrained)")
    print(f"Saved experiment artifacts: {output_dir}")


if __name__ == "__main__":
    main()
