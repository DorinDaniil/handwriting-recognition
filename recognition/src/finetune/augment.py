"""Light, recognition-safe augmentation for TrOCR fine-tuning.

Deliberately mild and glyph-preserving: no rotation, shear, perspective or padding (those
bend or pollute line crops). Only what bridges the gap to real scans + a real detector:
  - detector-style crop: trim each edge inward by a tiny fraction (width and height), so the
    model tolerates slightly-tight boxes — never enough to cut a word,
  - weak blur, JPEG artifacts, gentle colour/brightness shift.
"""
from __future__ import annotations

import io
import random

from PIL import Image
from torchvision.transforms import v2


class DetectorCropJitter:
    """Mimic a detector box that crops a hair tight: trim each edge inward by a small
    fraction of the side (width and height independently). Only ever shaves a sliver — no
    padding — so whole glyphs are kept."""

    def __init__(self, trim: tuple[float, float] = (0.01, 0.01), p: float = 0.8):
        self.trim, self.p = trim, p

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return image
        image = image.convert("RGB")
        w, h = image.size
        tw, th = self.trim
        left = round(random.uniform(0, tw) * w)
        right = w - round(random.uniform(0, tw) * w)
        top = round(random.uniform(0, th) * h)
        bottom = h - round(random.uniform(0, th) * h)
        return image.crop((left, top, max(left + 1, right), max(top + 1, bottom)))


class JpegArtifacts:
    """Re-encode through JPEG at a random quality to imitate scan/photo compression."""

    def __init__(self, quality: tuple[int, int] = (60, 95), p: float = 0.4):
        self.quality, self.p = quality, p

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return image
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=random.randint(*self.quality))
        buf.seek(0)
        return Image.open(buf).convert("RGB")


def build_transform() -> v2.Compose:
    return v2.Compose([
        DetectorCropJitter(trim=(0.01, 0.01), p=0.8),
        v2.RandomApply([v2.ColorJitter(brightness=0.12, contrast=0.12,
                                       saturation=0.06, hue=0.02)], p=0.5),
        v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))], p=0.3),
        JpegArtifacts(quality=(60, 95), p=0.4),
    ])


class Augmenter:
    def __init__(self, train: bool, p: float = 1.0):
        self.transform = build_transform() if train else None
        self.p = p                                   # chance to augment; 1-p -> clean image

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        if self.transform is None or random.random() >= self.p:
            return image
        return self.transform(image)
