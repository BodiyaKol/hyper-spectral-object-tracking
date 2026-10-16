"""Read HOT2026 false-color frames and their tracking annotations."""

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

BBox = tuple[float, float, float, float]


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
