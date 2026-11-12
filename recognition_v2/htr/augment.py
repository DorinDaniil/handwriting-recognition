"""Train-time augmentation of real line crops.

`add_context` puts a pre-cropped line back into page-like surroundings, because at inference
the crop is a detector box grown by a frame and brings real paper and neighbouring ink with it.
`Photometric` imitates scans and phone photos. Glyph geometry is never touched.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


def add_context(image: Image.Image, rng, pad_x, pad_y, donor: Image.Image | None = None) -> Image.Image:
    """Grow the canvas by random margins (in line heights). The margins continue the paper of the
    crop's own border; with a donor line, a sliver of it shows above or below as a neighbour."""
    line = np.asarray(image.convert("RGB"), dtype=np.float32)
    h, w = line.shape[:2]
    top, bottom = (int(h * rng.uniform(*pad_y)) for _ in range(2))
    left, right = (int(h * rng.uniform(*pad_x)) for _ in range(2))
    if not (top or bottom or left or right):
        return image

    stretched = np.pad(line, ((top, bottom), (left, right), (0, 0)), mode="edge")
    paper = Image.fromarray(stretched.astype(np.uint8)).filter(ImageFilter.GaussianBlur(max(2, h // 3)))
    canvas = np.asarray(paper, dtype=np.float32).copy()

    if donor is not None and (top or bottom):
        neighbour = np.asarray(donor.convert("RGB").resize(
            (max(1, round(donor.width * h / donor.height)), h), Image.BILINEAR), dtype=np.float32)
        shift = np.median(line.reshape(-1, 3), axis=0) - np.median(neighbour.reshape(-1, 3), axis=0)
        above = top and (not bottom or rng.random() < 0.5)
        room = top if above else bottom
        rows = min(room, h, max(2, int(room * rng.uniform(0.4, 1.0))))
        strip = neighbour[h - rows:] if above else neighbour[:rows]
        x0 = int(rng.integers(0, max(1, strip.shape[1] - canvas.shape[1] + 1)))
        strip = strip[:, x0:x0 + canvas.shape[1]] + shift
        at = int(rng.integers(0, canvas.shape[1] - strip.shape[1] + 1))
        y0 = 0 if above else canvas.shape[0] - rows
        area = canvas[y0:y0 + rows, at:at + strip.shape[1]]
        np.minimum(area, strip, out=area)

    canvas[top:top + h, left:left + w] = line
    return Image.fromarray(canvas.clip(0, 255).astype(np.uint8))


def shadow(image, rng):
    w, h = image.size
    mask = Image.new("L", (w, h), 0)
    corners = [(int(rng.integers(0, w + 1)), int(rng.integers(0, h + 1))) for _ in range(4)]
    ImageDraw.Draw(mask).polygon(corners, fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(max(3, w // 12)))
    darkness = np.asarray(mask, dtype=np.float32) / 255.0 * rng.uniform(0.15, 0.45)
    lit = np.asarray(image, dtype=np.float32) * (1.0 - darkness)[..., None]
    return Image.fromarray(lit.clip(0, 255).astype(np.uint8))


def colour(image, rng):
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.85, 1.15))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.85, 1.15))
    tinted = np.asarray(image, dtype=np.float32) * rng.uniform(0.9, 1.1, 3) + rng.uniform(-18, 18, 3)
    return Image.fromarray(tinted.clip(0, 255).astype(np.uint8))


def blur(image, rng):
    return image.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 1.0)))


def low_resolution(image, rng):
    w, h = image.size
    scale = rng.uniform(0.5, 0.85)
    small = image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def jpeg(image, rng):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=int(rng.integers(50, 96)))
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


class Photometric:
    OPS = ((shadow, 0.25), (colour, 0.5), (blur, 0.25), (low_resolution, 0.25), (jpeg, 0.4))

    def __init__(self, p: float):
        self.p = p

    def __call__(self, image: Image.Image, rng) -> Image.Image:
        image = image.convert("RGB")
        if rng.random() >= self.p:
            return image
        for op, p in self.OPS:
            if rng.random() < p:
                image = op(image, rng)
        return image
