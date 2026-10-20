import argparse
import sys
from pathlib import Path

import numpy as np

try:
	from hsi_tracking.data.dataset import load_hsi_sequence
	from hsi_tracking.data.x2cube import ChannelCountError, get_sensor_spec, select_bands, x2cube
except ModuleNotFoundError:  # running without `pip install -e .`
	sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
	from hsi_tracking.data.dataset import load_hsi_sequence
	from hsi_tracking.data.x2cube import ChannelCountError, get_sensor_spec, select_bands, x2cube


def describe(name: str, array: np.ndarray) -> str:
	"""One line with shape, channels, dtype, min/max and NaN/Inf counts."""
	channels = array.shape[2] if array.ndim == 3 else 1
	if np.issubdtype(array.dtype, np.floating):
		nan_count = int(np.isnan(array).sum())
		inf_count = int(np.isinf(array).sum())
	else:
		nan_count = inf_count = 0
	return (
		f"{name:<12} shape={array.shape} channels={channels} dtype={array.dtype} "
		f"min={array.min()} max={array.max()} nan={nan_count} inf={inf_count}"
	)


def main() -> int:
	parser = argparse.ArgumentParser(
		description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
	)
	parser.add_argument("--data", required=True, help="folder with HSI mosaic frames (PNG)")
	parser.add_argument("--sensor", required=True, help="VIS, NIR or RedNIR")
	parser.add_argument("--frames", type=int, default=5, help="how many first frames to check")
	parser.add_argument("--false-color", default=None, help="optional folder with the false-color pair")
	args = parser.parse_args()

	problems: list[str] = []

	try:
		spec = get_sensor_spec(args.sensor)
		sequence = load_hsi_sequence(args.data, spec.name, false_color_dir=args.false_color)
	except (ValueError, FileNotFoundError) as exc:
		print(f"ERROR: {exc}", file=sys.stderr)
		return 1

	print(
		f"Sensor: {spec.name} | block {spec.block_size}x{spec.block_size} | "
		f"raw bands {spec.raw_bands} -> expected {spec.expected_bands} "
		f"(dropped {spec.dropped_bands})"
	)
	print(f"Data: {args.data}")
	print(f"Frames: {len(sequence)} ({sequence.frame_paths[0].name} ... {sequence.frame_paths[-1].name})")

	if sequence.annotation_kind is None:
		print("Boxes: none found (no groundtruth_rect.txt / init_rect.txt)")
	else:
		print(f"Boxes: {len(sequence.boxes)} from {sequence.annotation_kind} file, first = {sequence.boxes[0]}")
		if sequence.annotation_kind == "groundtruth":
			print(f"Boxes vs frames: {len(sequence.boxes)} / {len(sequence)} -> OK")
		else:
			print("Boxes vs frames: only the first frame is annotated (init_rect.txt)")

	if sequence.false_color_paths is not None:
		print(f"False-color: {len(sequence.false_color_paths)} frames, frame numbers match HSI -> OK")

	count = min(args.frames, len(sequence))
	print(f"\nChecking first {count} frame(s):")
	for index in range(count):
		print(f"\n[{index}] {sequence.frame_paths[index].name}")
		try:
			mosaic = sequence.load_mosaic(index)
			print("  " + describe("before:", mosaic))

			leftover = (mosaic.shape[0] % spec.block_size, mosaic.shape[1] % spec.block_size)
			if leftover != (0, 0):
				print(
					f"  note: mosaic is not a multiple of {spec.block_size}, "
					f"X2Cube drops {leftover[0]} row(s) and {leftover[1]} column(s)"
				)

			raw_cube = x2cube(mosaic, spec.block_size)
			print("  " + describe("raw cube:", raw_cube))
			cube = select_bands(raw_cube, spec.name)
			if spec.dropped_bands:
				dropped = raw_cube[..., spec.expected_bands:]
				status = "OK (all zero)" if not dropped.any() else "WARNING: not zero"
				print(f"  dropped band(s) max={dropped.max()} -> {status}")
				if dropped.any():
					problems.append(f"frame {index}: dropped {spec.name} band is not zero")
			print("  " + describe("after:", cube))

			sample = sequence[index]
			print("  " + describe("loader:", sample.hsi))
			if sample.hsi.shape != cube.shape:
				problems.append(f"frame {index}: loader shape {sample.hsi.shape} != {cube.shape}")
			print(f"  bbox: {sample.bbox}")

			if sample.false_color is not None:
				fc_size = sample.false_color.shape[:2]
				hsi_size = sample.hsi.shape[:2]
				ok = fc_size == hsi_size
				print(f"  false-color size {fc_size} vs HSI {hsi_size} -> {'OK' if ok else 'MISMATCH'}")
				if not ok:
					problems.append(f"frame {index}: false-color {fc_size} != HSI {hsi_size}")
		except ChannelCountError as exc:
			print(f"\nERROR: wrong number of channels: {exc}", file=sys.stderr)
			return 1
		except ValueError as exc:
			print(f"\nERROR: {exc}", file=sys.stderr)
			return 1

	print()
	if problems:
		print(f"Found {len(problems)} problem(s):")
		for problem in problems:
			print(f"  - {problem}")
		return 1
	print("All checks passed.")
	return 0


if __name__ == "__main__":
	sys.exit(main())
