from PIL import Image

from hsi_tracking.visualization import save_initial_frame_preview


def test_save_initial_frame_preview_draws_init_box(tmp_path) -> None:
    sequence_dir = tmp_path / "doll3"
    sequence_dir.mkdir()
    Image.new("RGB", (400, 300), "white").save(sequence_dir / "0002.jpg")
    Image.new("RGB", (400, 300), "white").save(sequence_dir / "0001.jpg")
    (sequence_dir / "init_rect.txt").write_text("253.0 150.0 58.0 71.0", encoding="utf-8")

    output_path = save_initial_frame_preview(sequence_dir, tmp_path / "preview.png")

    with Image.open(output_path) as preview:
        assert preview.size == (400, 300)
        assert preview.getpixel((253, 150)) == (255, 0, 0)
