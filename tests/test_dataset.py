import csv

import numpy as np
import pytest
from PIL import Image

from hsi_tracking.data.dataset import load_hsi_sequence, load_tracking_sequence, read_bbox_file
from hsi_tracking.data.x2cube import (
    SENSORS,
    ChannelCountError,
    load_mosaic,
    mosaic_to_cube,
    select_bands,
    validate_channels,
    x2cube,
)



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



def official_x2cube(img, B=[4, 4], skip=[4, 4], bandNumber=16):
    """Verbatim copy of X2Cube from the organizers' HyperTools.py (used as a reference)."""
    M, N = img.shape  # noqa: N806
    col_extent = N - B[1] + 1
    row_extent = M - B[0] + 1
    start_idx = np.arange(B[0])[:, None] * N + np.arange(B[1])
    didx = M * N * np.arange(1)
    start_idx = (didx[:, None] + start_idx.ravel()).reshape((-1, B[0], B[1]))
    offset_idx = np.arange(row_extent)[:, None] * N + np.arange(col_extent)
    out = np.take(img, start_idx.ravel()[:, None] + offset_idx[::skip[0], ::skip[1]].ravel())
    out = np.transpose(out)
    DataCube = out.reshape(M // 4, N // 4, bandNumber)  # noqa: N806
    return DataCube


def official_x2cube_any_block(img, block):
    """Official X2Cube with the hard-coded 4 in the last reshape replaced by the block size.

    The original only works for 4 x 4 mosaics (VIS, RedNIR); NIR uses 5 x 5 blocks.
    """
    M, N = img.shape  # noqa: N806
    start_idx = np.arange(block)[:, None] * N + np.arange(block)
    offset_idx = np.arange(M - block + 1)[:, None] * N + np.arange(N - block + 1)
    out = np.take(img, start_idx.ravel()[:, None] + offset_idx[::block, ::block].ravel())
    return np.transpose(out).reshape(M // block, N // block, block * block)


def make_mosaic(sensor, rows=6, cols=8, zero_last_band=True, seed=0):
    """Random mosaic for a sensor; the RedNIR last band is zero like on the real camera."""
    spec = SENSORS[sensor]
    block = spec.block_size
    rng = np.random.default_rng(seed)
    mosaic = rng.integers(1, 1000, size=(rows * block, cols * block), dtype=np.uint16)
    if spec.dropped_bands and zero_last_band:
        mosaic[block - 1 :: block, block - 1 :: block] = 0
    return mosaic


def write_sequence(folder, sensor, n_frames=3, boxes_text=None, rows=6, cols=8):
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(1, n_frames + 1):
        Image.fromarray(make_mosaic(sensor, rows, cols, seed=i)).save(folder / f"{i:04d}.png")
    if boxes_text is not None:
        (folder / "groundtruth_rect.txt").write_text(boxes_text)
    return folder



def test_x2cube_matches_official_code_vis() -> None:
    mosaic = make_mosaic("VIS")
    np.testing.assert_array_equal(x2cube(mosaic, 4), official_x2cube(mosaic))


def test_x2cube_matches_official_code_nir() -> None:
    mosaic = make_mosaic("NIR")
    np.testing.assert_array_equal(x2cube(mosaic, 5), official_x2cube_any_block(mosaic, 5))


def test_official_code_fails_for_nir_blocks() -> None:
    # documents why x2cube does not reuse the official function as is
    with pytest.raises(ValueError):
        official_x2cube(make_mosaic("NIR"), B=[5, 5], skip=[5, 5], bandNumber=25)


def test_x2cube_band_order() -> None:
    # every pixel of band k has value k
    block = 4
    band_of_pixel = np.arange(block)[:, None] * block + np.arange(block)
    mosaic = np.tile(band_of_pixel, (3, 5))
    cube = x2cube(mosaic, block)
    assert cube.shape == (3, 5, 16)
    for k in range(16):
        assert (cube[..., k] == k).all()


def test_x2cube_drops_incomplete_blocks_like_official() -> None:
    mosaic = np.arange(23 * 18, dtype=np.uint16).reshape(23, 18)
    cube = x2cube(mosaic, 5)
    assert cube.shape == (4, 3, 25)
    np.testing.assert_array_equal(cube, official_x2cube_any_block(mosaic, 5))


def test_x2cube_rejects_non_2d_input() -> None:
    with pytest.raises(ValueError):
        x2cube(np.zeros((8, 8, 3)), 4)



@pytest.mark.parametrize(("sensor", "bands"), [("VIS", 16), ("NIR", 25), ("RedNIR", 15)])
def test_mosaic_to_cube_channel_count(sensor, bands) -> None:
    cube = mosaic_to_cube(make_mosaic(sensor, rows=6, cols=8), sensor)
    assert cube.shape == (6, 8, bands)


def test_rednir_removes_last_band() -> None:
    mosaic = make_mosaic("RedNIR")
    raw = x2cube(mosaic, 4)
    assert not raw[..., 15].any()  # last raw band is zero
    np.testing.assert_array_equal(mosaic_to_cube(mosaic, "RedNIR"), raw[..., :15])


def test_sensor_name_is_case_insensitive() -> None:
    assert mosaic_to_cube(make_mosaic("NIR"), "nir").shape[2] == 25


def test_wrong_channel_count_raises() -> None:
    with pytest.raises(ChannelCountError):
        validate_channels(np.zeros((4, 4, 16)), "NIR")
    with pytest.raises(ChannelCountError):
        select_bands(np.zeros((4, 4, 25)), "VIS")


def test_unknown_sensor_raises() -> None:
    with pytest.raises(ValueError):
        mosaic_to_cube(make_mosaic("VIS"), "SWIR")



def test_rgb_frame_is_not_accepted_as_mosaic(tmp_path) -> None:
    path = tmp_path / "0001.png"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(path)
    with pytest.raises(ValueError, match="not a single-channel mosaic"):
        load_mosaic(path)


def test_corrupted_frame_raises(tmp_path) -> None:
    path = tmp_path / "0001.png"
    path.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="Cannot read"):
        load_mosaic(path)


def test_missing_folder_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_hsi_sequence(tmp_path / "nope", "VIS")


def test_folder_without_mosaics_raises(tmp_path) -> None:
    (tmp_path / "0001.jpg").write_bytes(b"")
    with pytest.raises(ValueError, match="No HSI mosaic frames"):
        load_hsi_sequence(tmp_path, "VIS")



def test_bbox_file_formats(tmp_path) -> None:
    path = tmp_path / "groundtruth_rect.txt"
    path.write_text("214\t61\t19\t21\t\n253.0 150.0 58.0 71.0\r\n1,2,3,4\n\n")
    assert read_bbox_file(path) == [(214, 61, 19, 21), (253, 150, 58, 71), (1, 2, 3, 4)]


def test_bad_bbox_line_raises(tmp_path) -> None:
    path = tmp_path / "groundtruth_rect.txt"
    path.write_text("1 2 3\n")
    with pytest.raises(ValueError, match="expected 4 numbers"):
        read_bbox_file(path)


def test_bbox_count_mismatch_raises(tmp_path) -> None:
    folder = write_sequence(tmp_path / "seq", "VIS", n_frames=3, boxes_text="1 2 3 4\n5 6 7 8\n")
    with pytest.raises(ValueError, match="count mismatch"):
        load_hsi_sequence(folder, "VIS")



def test_load_hsi_sequence_returns_hwc_bbox_and_sensor(tmp_path) -> None:
    boxes = "10\t20\t30\t40\t\n11\t21\t30\t40\t\n12\t22\t30\t40\t\n"
    folder = write_sequence(tmp_path / "seq", "NIR", n_frames=3, boxes_text=boxes)
    sequence = load_hsi_sequence(folder, "NIR")
    sample = sequence[1]
    assert len(sequence) == 3
    assert sample.hsi.shape == (6, 8, 25)
    assert sample.hsi.dtype == np.float32
    assert sample.bbox == (11, 21, 30, 40)
    assert sample.sensor == "NIR"
    assert sample.false_color is None


def test_init_rect_annotates_only_first_frame(tmp_path) -> None:
    folder = write_sequence(tmp_path / "seq", "VIS", n_frames=3)
    (folder / "init_rect.txt").write_text("253.0 150.0 58.0 71.0\r\n")
    sequence = load_hsi_sequence(folder, "VIS")
    assert sequence.annotation_kind == "init"
    assert sequence[0].bbox == (253, 150, 58, 71)
    assert sequence[2].bbox is None


def test_false_color_pair_is_loaded_separately(tmp_path) -> None:
    folder = write_sequence(tmp_path / "hsi", "VIS", n_frames=2)
    fc = tmp_path / "fc"
    fc.mkdir()
    for i in (1, 2):
        Image.fromarray(np.full((6, 8, 3), 255, dtype=np.uint8)).save(fc / f"{i:04d}.jpg")
    sample = load_hsi_sequence(folder, "VIS", false_color_dir=fc)[0]
    assert sample.hsi.shape == (6, 8, 16)
    assert sample.false_color.shape == (6, 8, 3)
    assert sample.false_color.max() <= 1.0


def test_false_color_count_mismatch_raises(tmp_path) -> None:
    folder = write_sequence(tmp_path / "hsi", "VIS", n_frames=2)
    fc = tmp_path / "fc"
    fc.mkdir()
    Image.fromarray(np.zeros((6, 8, 3), dtype=np.uint8)).save(fc / "0001.jpg")
    with pytest.raises(ValueError, match="frame count mismatch"):
        load_hsi_sequence(folder, "VIS", false_color_dir=fc)
