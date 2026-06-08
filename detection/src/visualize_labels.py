from pathlib import Path
import json
import colorsys
from PIL import Image, ImageDraw, ImageFont


def _color_for(i: int, n: int) -> tuple[int, int, int]:
    h = i / max(n, 1)
    r, g, b = colorsys.hsv_to_rgb(h, 0.9, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def _load_font(size: int = 42):
    return ImageFont.load_default(size=size)


def render_boxes(
    image_path: Path | str,
    boxes: list[dict],
    line_width: int = 3,
    font_size: int = 42,
) -> Image.Image:
    """Draw polygons + scores on image, return the PIL.Image (not saved)."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _load_font(font_size)

    for i, b in enumerate(boxes):
        pts = [(float(x), float(y)) for x, y in b["points"]]
        color = _color_for(i, len(boxes))
        draw.polygon(pts, outline=color, width=line_width)

        score = b.get("score")
        if score is None:
            continue
        label = f"{score:.2f}"
        x, y = pts[0]
        tx, ty = x, max(0, y - font_size - 4)
        bbox = draw.textbbox((tx, ty), label, font=font)
        draw.rectangle(bbox, fill=color)
        draw.text((tx, ty), label, fill="black", font=font)

    return img


def show_from_labels(
    labels_txt: Path | str,
    dataset_root: Path | str,
    idx: int | None = None,
    rel_path: str | None = None,
    line_width: int = 3,
    font_size: int = 42,
) -> Image.Image:
    """
    Отрисовать одну картинку из labels.txt. Вернуть PIL.Image.

    Args:
        labels_txt:   путь к labels.txt
        dataset_root: корень датасета (относительно него пути в labels.txt)
        idx:          номер строки в labels.txt (0-based). Используется если rel_path=None.
        rel_path:     относительный путь из labels.txt — альтернатива idx.
    """
    labels_txt = Path(labels_txt)
    dataset_root = Path(dataset_root)

    with open(labels_txt, encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]

    target = None
    if rel_path is not None:
        for line in lines:
            rel, payload = line.split("\t", 1)
            if rel == rel_path:
                target = (rel, payload)
                break
        if target is None:
            raise ValueError(f"rel_path {rel_path!r} not found in {labels_txt}")
    else:
        if idx is None:
            raise ValueError("Provide either idx or rel_path")
        if not (0 <= idx < len(lines)):
            raise IndexError(f"idx {idx} out of range [0, {len(lines)})")
        target = lines[idx].split("\t", 1)

    rel, payload = target
    boxes = json.loads(payload)
    print(f"{rel}  ({len(boxes)} boxes)")

    return render_boxes(
        dataset_root / rel,
        boxes,
        line_width=line_width,
        font_size=font_size,
    )