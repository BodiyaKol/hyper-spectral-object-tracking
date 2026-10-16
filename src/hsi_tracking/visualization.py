"""Utilities for rendering tracking predictions and initialization previews."""

from pathlib import Path

from PIL import Image, ImageDraw


def save_initial_frame_preview(
    sequence_dir: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Draw ``init_rect.txt`` on the first image and save the preview as PNG."""
    sequence_dir = Path(sequence_dir)
    frame_paths = sorted(
        (
            path
            for path in sequence_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ),
        key=lambda path: (
            not path.stem.isdigit(),
            int(path.stem) if path.stem.isdigit() else path.name,
        ),
    )
    if not frame_paths:
        raise FileNotFoundError(f"No image frames found in {sequence_dir}")

    init_rect_path = sequence_dir / "init_rect.txt"
    values = init_rect_path.read_text(encoding="utf-8").replace(",", " ").split()
    if len(values) != 4:
        raise ValueError(f"Expected x y width height in {init_rect_path}")
    x, y, width, height = map(float, values)
    if width <= 0 or height <= 0:
        raise ValueError(f"Bounding-box width and height must be positive: {values}")

    with Image.open(frame_paths[0]) as source:
        image = source.convert("RGB")
    right, bottom = x + width, y + height
    if x < 0 or y < 0 or right > image.width or bottom > image.height:
        raise ValueError(f"Bounding box {values} falls outside the first frame {image.size}")

    draw = ImageDraw.Draw(image)
    draw.rectangle((round(x), round(y), round(right), round(bottom)), outline="red", width=3)

    output_path = Path(output_path or f"results/visualizations/{sequence_dir.name}_init.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path
