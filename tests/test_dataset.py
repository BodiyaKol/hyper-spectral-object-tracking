import csv

import numpy as np
from PIL import Image

from hsi_tracking.data.dataset import load_tracking_sequence


def test_load_tracking_sequence_sorts_frames_and_preprocesses_rgb(tmp_path) -> None:
    video_dir = tmp_path / "automobile4"
    video_dir.mkdir()
    Image.fromarray(np.full((4, 5, 3), 255, dtype=np.uint8)).save(video_dir / "0002.jpg")
    Image.fromarray(np.zeros((4, 5, 3), dtype=np.uint8)).save(video_dir / "0001.jpg")

    annotations_path = tmp_path / "training.csv"
    with annotations_path.open("w", newline="", encoding="utf-8") as annotations_file:
        writer = csv.DictWriter(annotations_file, fieldnames=["ID", "x", "y", "width", "height"])
        writer.writeheader()
        writer.writerow({"ID": "other-video_1", "x": 9, "y": 9, "width": 9, "height": 9})
        writer.writerow({"ID": "vis-automobile4_2", "x": 5, "y": 6, "width": 7, "height": 8})
        writer.writerow({"ID": "vis-automobile4_1", "x": 1, "y": 2, "width": 3, "height": 4})

    sequence = load_tracking_sequence(video_dir, annotations_path, "vis-automobile4")

    frame, box = sequence[0]
    assert len(sequence) == 2
    assert sequence.frame_paths[0].name == "0001.jpg"
    assert frame.shape == (3, 4, 5)
    assert frame.dtype == np.float32
    assert frame.min() == 0.0
    assert box == (1.0, 2.0, 3.0, 4.0)


def test_load_tracking_sequence_rejects_missing_annotations(tmp_path) -> None:
    video_dir = tmp_path / "automobile4"
    video_dir.mkdir()
    Image.fromarray(np.zeros((4, 5, 3), dtype=np.uint8)).save(video_dir / "0001.jpg")
    annotations_path = tmp_path / "training.csv"
    annotations_path.write_text("ID,x,y,width,height\n", encoding="utf-8")

    try:
        load_tracking_sequence(video_dir, annotations_path, "vis-automobile4")
    except ValueError as error:
        assert "No annotations found" in str(error)
    else:
        raise AssertionError("Expected missing annotations to be rejected")
