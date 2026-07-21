"""Photometric augmentation for the essay fine-tune (geometry/frame jitter lives in the
dataset, since it needs the full page). Everything here operates PIL->PIL and stays light:
soft shadows, gentle colour shift, weak blur, a quality cut (downscale) and JPEG artifacts.
The JPEG re-encoder is reused from the main fine-tune augmenter — not re-implemented.
"""
from __future__ import annotations

import random

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


class RandomShadow:
    """Darken a soft random quadrilateral region — imitates page shadows / uneven lighting."""

    def __init__(self, p: float = 0.3, darkness: tuple[float, float] = (0.15, 0.45)):
        self.p, self.darkness = p, darkness

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return image
        w, h = image.size
        mask = Image.new("L", (w, h), 0)
        pts = [(random.randint(0, w), random.randint(0, h)) for _ in range(4)]
        ImageDraw.Draw(mask).polygon(pts, fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=max(3, w // 12)))
        m = np.asarray(mask, dtype=np.float32) / 255.0 * random.uniform(*self.darkness)
        arr = np.asarray(image, dtype=np.float32) * (1.0 - m)[..., None]
        return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


class ColorShift:
    """Brightness / contrast + an explicit colour cast. The cast is a per-channel gain & bias plus
    a hue rotation, so it visibly tints even a black-on-white scan (paper and ink take on a colour)
    — the point is to stop the recognizer overfitting to clean grey scans."""

    def __init__(self, p: float = 0.6, brightness: float = 0.15, contrast: float = 0.15,
                 tint_gain: float = 0.12, tint_bias: float = 22.0, hue: float = 0.06):
        self.p, self.b, self.c = p, brightness, contrast
        self.tg, self.tb, self.hue = tint_gain, tint_bias, hue

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return image
        if self.b:
            image = ImageEnhance.Brightness(image).enhance(1.0 + random.uniform(-self.b, self.b))
        if self.c:
            image = ImageEnhance.Contrast(image).enhance(1.0 + random.uniform(-self.c, self.c))
        # per-channel colour cast — introduces colour where a grey scan has none
        if self.tg or self.tb:
            arr = np.asarray(image, dtype=np.float32)
            gain = 1.0 + np.random.uniform(-self.tg, self.tg, 3).astype(np.float32)
            bias = np.random.uniform(-self.tb, self.tb, 3).astype(np.float32)
            image = Image.fromarray((arr * gain + bias).clip(0, 255).astype(np.uint8))
        # hue rotation on the now-tinted image
        if self.hue:
            hsv = np.asarray(image.convert("HSV"), dtype=np.int16)
            hsv[..., 0] = (hsv[..., 0] + int(random.uniform(-self.hue, self.hue) * 255)) % 256
            image = Image.fromarray(hsv.astype(np.uint8), "HSV").convert("RGB")
        return image


class WeakBlur:
    def __init__(self, p: float = 0.3, radius: tuple[float, float] = (0.1, 0.8)):
        self.p, self.radius = p, radius

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return image
        return image.filter(ImageFilter.GaussianBlur(radius=random.uniform(*self.radius)))


class QualityDrop:
    """Downscale then upscale back — imitates a low-resolution scan / detector crop."""

    def __init__(self, p: float = 0.3, scale: tuple[float, float] = (0.5, 0.85)):
        self.p, self.scale = p, scale

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return image
        w, h = image.size
        s = random.uniform(*self.scale)
        small = image.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)


class EssayAugmenter:
    """Applies the photometric pipeline with overall probability `p` (else a clean crop)."""

    def __init__(self, train: bool, p: float = 0.7):
        self.p = p
        if not train:
            self.ops = []
            return
        from src.finetune.augment import JpegArtifacts   # reuse (lazy: pulls torchvision)
        self.ops = [RandomShadow(p=0.3), ColorShift(p=0.5), WeakBlur(p=0.3),
                    QualityDrop(p=0.3), JpegArtifacts(quality=(55, 95), p=0.4)]

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        if not self.ops or random.random() >= self.p:
            return image
        for op in self.ops:
            image = op(image)
        return image
