"""Line image -> model input: fixed height, width that follows the aspect ratio."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

WIDTH_STEP = 32


def fit_height(image: Image.Image, height: int, max_width: int) -> np.ndarray:
    """Resize to `height` keeping the aspect ratio; only lines wider than max_width get squeezed."""
    w, h = image.size
    width = min(max_width, max(height // 4, round(w * height / max(h, 1))))
    return np.array(image.convert("RGB").resize((width, height), Image.BILINEAR))


def to_batch(arrays: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    """uint8 [H, W, 3] images of one height -> float [B, 3, H, W_max] in [-1, 1] and their widths."""
    widths = torch.tensor([a.shape[1] for a in arrays])
    padded = -(-int(widths.max()) // WIDTH_STEP) * WIDTH_STEP
    batch = torch.zeros(len(arrays), 3, arrays[0].shape[0], padded)
    for i, a in enumerate(arrays):
        batch[i, :, :, :a.shape[1]] = torch.from_numpy(a).permute(2, 0, 1) / 127.5 - 1.0
    return batch, widths
