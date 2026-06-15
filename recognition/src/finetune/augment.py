from __future__ import annotations

import io
import random

import numpy as np
from PIL import Image
from torchvision.transforms import v2

FILL = 255


class GaussianNoise:
    def __init__(self, std=(5.0, 20.0), p=0.3):
        self.std = std
        self.p = p

    def __call__(self, image):
        if random.random() >= self.p:
            return image
        arr = np.asarray(image, dtype=np.float32)
        arr += np.random.normal(0.0, random.uniform(*self.std), arr.shape)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


class JpegCompression:
    def __init__(self, quality=(50, 95), p=0.3):
        self.quality = quality
        self.p = p

    def __call__(self, image):
        if random.random() >= self.p:
            return image
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=random.randint(*self.quality))
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


def build_transform():
    return v2.Compose([
        v2.RandomAffine(degrees=3, shear=4, scale=(0.9, 1.1), fill=FILL),
        v2.RandomPerspective(distortion_scale=0.2, p=0.3, fill=FILL),
        v2.ColorJitter(brightness=0.2, contrast=0.2),
        v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.2),
        GaussianNoise(std=(5.0, 20.0), p=0.3),
        JpegCompression(quality=(50, 95), p=0.3),
    ])


class Augmenter:
    def __init__(self, train: bool):
        self.transform = build_transform() if train else None

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        return self.transform(image) if self.transform is not None else image
