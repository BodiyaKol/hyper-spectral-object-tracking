from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple  # noqa: UP035

import numpy as np
from PIL import Image

from hsi_tracking.data.x2cube import get_sensor_spec, load_mosaic, mosaic_to_cube

BBox = Tuple[float, float, float, float]  # noqa: UP006


@dataclass
class TrackingSequence:
	"""A lazily loaded video sequence aligned with its ground-truth boxes."""

	frame_paths: list[Path]
	boxes: list[BBox]

	def __len__(self) -> int:
		return len(self.frame_paths)

	def __getitem__(self, index: int) -> tuple[np.ndarray, BBox]:
		"""Return one RGB frame as float32 CHW in [0, 1] and its (x, y, w, h) box."""
		with Image.open(self.frame_paths[index]) as image:
			frame = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
		frame = np.ascontiguousarray(frame.transpose(2, 0, 1))
		return frame, self.boxes[index]


def _frame_sort_key(path: Path) -> tuple[int, int | str]:
	match = re.search(r"\d+", path.stem)
	if match:
		return 0, int(match.group())
	return 1, path.name.casefold()


def load_tracking_sequence(
	video_dir: str | Path,
	annotations_csv: str | Path,
	video_id: str,
) -> TrackingSequence:
	"""Load sorted JPG paths and matching annotations for one video ID.

	The CSV is expected to contain ``ID,x,y,width,height`` columns, where IDs
	follow the ``<video_id>_<index>`` format used by the HOT2026 training CSV.
	"""
	video_dir = Path(video_dir)
	csv_path = Path(annotations_csv)
	if not video_dir.is_dir():
		raise FileNotFoundError(f"Video directory does not exist: {video_dir}")
	if not csv_path.is_file():
		raise FileNotFoundError(f"Annotations CSV does not exist: {csv_path}")

	frame_paths = sorted(
		(path for path in video_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg"}),
		key=_frame_sort_key,
	)
	if not frame_paths:
		raise ValueError(f"No JPG frames found in {video_dir}")

	boxes_with_index: list[tuple[int, BBox]] = []
	with csv_path.open(encoding="utf-8-sig", newline="") as annotations_file:
		reader = csv.DictReader(annotations_file)
		required_columns = {"ID", "x", "y", "width", "height"}
		if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
			raise ValueError(f"Annotations CSV must contain columns: {sorted(required_columns)}")

		for row in reader:
			row_video_id, separator, frame_index = row["ID"].rpartition("_")
			if not separator or row_video_id != video_id:
				continue
			boxes_with_index.append(
				(
					int(frame_index),
					(float(row["x"]), float(row["y"]), float(row["width"]), float(row["height"])),
				)
			)

	boxes_with_index.sort(key=lambda item: item[0])
	boxes = [box for _, box in boxes_with_index]
	if not boxes:
		raise ValueError(f"No annotations found for video ID {video_id!r} in {csv_path}")
	if len(frame_paths) != len(boxes):
		raise ValueError(
			f"Frame/annotation count mismatch for {video_id!r}: "
			f"{len(frame_paths)} frames, {len(boxes)} boxes"
		)

	return TrackingSequence(frame_paths=frame_paths, boxes=boxes)



HSI_EXTENSIONS = {".png", ".tif", ".tiff"}
FALSE_COLOR_EXTENSIONS = {".jpg", ".jpeg", ".png"}
GROUNDTRUTH_FILE = "groundtruth_rect.txt"  # boxes for every frame
INIT_RECT_FILE = "init_rect.txt"  # box for the first frame only


def read_bbox_file(path: str | Path) -> list[BBox]:
	"""Read (x, y, width, height) boxes, one per line.

	Numbers may be separated by tabs, spaces or commas; trailing separators,
	Windows line endings and float values (e.g. "253.0") are accepted.
	A line that does not contain exactly 4 numbers raises ValueError.
	"""
	path = Path(path)
	boxes: list[BBox] = []
	for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
		stripped = line.strip()
		if not stripped:
			continue
		parts = [part for part in re.split(r"[,\s]+", stripped) if part]
		if len(parts) != 4:
			raise ValueError(
				f"{path}:{line_number}: expected 4 numbers (x, y, width, height), got {line!r}"
			)
		try:
			x, y, width, height = (float(part) for part in parts)
		except ValueError as exc:
			raise ValueError(f"{path}:{line_number}: not a number in {line!r}") from exc
		boxes.append((x, y, width, height))

	if not boxes:
		raise ValueError(f"No boxes found in {path}")
	return boxes


def _list_frames(directory: Path, extensions: set[str]) -> list[Path]:
	return sorted(
		(path for path in directory.iterdir() if path.suffix.lower() in extensions),
		key=_frame_sort_key,
	)


def _find_annotations(*directories: Path) -> tuple[list[BBox], str | None]:
	"""Return boxes and their kind ("groundtruth", "init" or None if there are none)."""
	for directory in directories:
		if (directory / GROUNDTRUTH_FILE).is_file():
			return read_bbox_file(directory / GROUNDTRUTH_FILE), "groundtruth"
		if (directory / INIT_RECT_FILE).is_file():
			return read_bbox_file(directory / INIT_RECT_FILE), "init"
	return [], None


def _read_false_color(path: Path) -> np.ndarray:
	try:
		with Image.open(path) as image:
			return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
	except OSError as exc:
		raise ValueError(f"Cannot read false-color frame {path}: {exc}") from exc


@dataclass
class HSISample:
	"""One frame of an HSI sequence.

	hsi:         H x W x C float32 cube, raw sensor intensities (not rescaled).
	bbox:        (x, y, width, height), or None if this frame has no annotation.
	sensor:      "VIS", "NIR" or "RedNIR".
	false_color: H x W x 3 float32 in [0, 1], read separately from the HSI cube
		(None if no false-color folder was given).
	"""

	hsi: np.ndarray
	bbox: BBox | None
	sensor: str
	frame_path: Path
	false_color: np.ndarray | None = None


@dataclass
class HSISequence:
	"""A lazily loaded hyperspectral video: mosaic frames + boxes + sensor info."""

	sensor: str
	frame_paths: list[Path]
	boxes: list[BBox]
	annotation_kind: str | None
	false_color_paths: list[Path] | None = None

	def __len__(self) -> int:
		return len(self.frame_paths)

	def bbox(self, index: int) -> BBox | None:
		return self.boxes[index] if index < len(self.boxes) else None

	def load_mosaic(self, index: int) -> np.ndarray:
		"""Raw 2D mosaic exactly as stored on disk (before X2Cube)."""
		return load_mosaic(self.frame_paths[index])

	def load_cube(self, index: int) -> np.ndarray:
		"""H x W x C float32 cube after X2Cube and band selection."""
		return mosaic_to_cube(self.load_mosaic(index), self.sensor).astype(np.float32)

	def load_false_color(self, index: int) -> np.ndarray | None:
		if self.false_color_paths is None:
			return None
		return _read_false_color(self.false_color_paths[index])

	def __getitem__(self, index: int) -> HSISample:
		return HSISample(
			hsi=self.load_cube(index),
			bbox=self.bbox(index),
			sensor=self.sensor,
			frame_path=self.frame_paths[index],
			false_color=self.load_false_color(index),
		)


def load_hsi_sequence(
	sequence_dir: str | Path,
	sensor: str,
	false_color_dir: str | Path | None = None,
) -> HSISequence:
	"""Load one HSI sequence folder (mosaic PNG frames + optional boxes).

	Boxes are taken from groundtruth_rect.txt (all frames) or init_rect.txt
	(first frame only), looked up in the HSI folder first, then in the
	false-color folder. If false_color_dir is given, its frames must have the
	same count and frame numbers as the HSI frames.
	"""
	spec = get_sensor_spec(sensor)
	sequence_dir = Path(sequence_dir)
	if not sequence_dir.is_dir():
		raise FileNotFoundError(f"HSI sequence directory does not exist: {sequence_dir}")

	frame_paths = _list_frames(sequence_dir, HSI_EXTENSIONS)
	if not frame_paths:
		raise ValueError(
			f"No HSI mosaic frames ({', '.join(sorted(HSI_EXTENSIONS))}) found in {sequence_dir}"
		)

	false_color_paths = None
	search_dirs = [sequence_dir]
	if false_color_dir is not None:
		false_color_dir = Path(false_color_dir)
		if not false_color_dir.is_dir():
			raise FileNotFoundError(f"False-color directory does not exist: {false_color_dir}")
		false_color_paths = _list_frames(false_color_dir, FALSE_COLOR_EXTENSIONS)
		if len(false_color_paths) != len(frame_paths):
			raise ValueError(
				f"HSI/false-color frame count mismatch: {len(frame_paths)} HSI frames, "
				f"{len(false_color_paths)} false-color frames"
			)
		hsi_numbers = [_frame_sort_key(path) for path in frame_paths]
		fc_numbers = [_frame_sort_key(path) for path in false_color_paths]
		if hsi_numbers != fc_numbers:
			raise ValueError("HSI and false-color frame numbers do not match")
		search_dirs.append(false_color_dir)

	boxes, annotation_kind = _find_annotations(*search_dirs)
	if annotation_kind == "groundtruth" and len(boxes) != len(frame_paths):
		raise ValueError(
			f"Frame/annotation count mismatch in {sequence_dir}: "
			f"{len(frame_paths)} frames, {len(boxes)} boxes"
		)

	return HSISequence(
		sensor=spec.name,
		frame_paths=frame_paths,
		boxes=boxes,
		annotation_kind=annotation_kind,
		false_color_paths=false_color_paths,
	)
