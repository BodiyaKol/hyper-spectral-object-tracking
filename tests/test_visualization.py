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


def test_tracking_metrics_plots(tmp_path):
    from hsi_tracking.visualization import plot_tracking_metrics

    gt = [
        [100, 100, 50, 50],
        [110, 100, 50, 50],
        [120, 100, 50, 50],
    ]

    pred = [
        [100, 100, 50, 50],
        [115, 100, 50, 50],
        [130, 100, 50, 50],
    ]

    success_path, precision_path = plot_tracking_metrics(
        ground_truth_boxes=gt,
        predicted_boxes=pred,
        model_name="Test Model",
        output_dir=tmp_path,
    )

    assert success_path.exists()
    assert precision_path.exists()
    assert success_path.stat().st_size > 0
    assert precision_path.stat().st_size > 0

