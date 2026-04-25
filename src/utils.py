"""Small utilities: seeding, preprocessing, visualization, detection H-mean metric, EMA."""
from __future__ import annotations

import colorsys
import copy
import logging
import os
import random

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --- reproducibility ------------------------------------------------------

def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


# --- visualization --------------------------------------------------------

def _color_for(i: int, n: int) -> tuple[int, int, int]:
    h = i / max(n, 1)
    r, g, b = colorsys.hsv_to_rgb(h, 0.9, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def draw_polygons(
    image: np.ndarray | Image.Image,
    polygons: list[np.ndarray],
    scores: list[float] | None = None,
    thickness: int = 2,
) -> np.ndarray:
    """Draw polygons + optional scores on an RGB image using PIL (no cv2).

    Accepts a (H, W, 3) uint8 numpy array or a PIL.Image.Image. Returns a
    new (H, W, 3) uint8 numpy array.
    """
    pil = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image))
    pil = pil.convert("RGB").copy()
    draw = ImageDraw.Draw(pil)
    n = len(polygons)
    for i, poly in enumerate(polygons):
        pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
        if len(pts) < 2:
            continue
        color = _color_for(i, n)
        # draw the closed polyline with a thick stroke
        xy = [(float(x), float(y)) for x, y in pts]
        xy.append(xy[0])
        draw.line(xy, fill=color, width=thickness)
        if scores is not None:
            tx = int(round(pts[0, 0]))
            ty = int(round(pts[0, 1])) - 12
            draw.text((tx, max(0, ty)), f"{scores[i]:.2f}", fill=color)
    return np.asarray(pil)


def denormalize(image_chw: torch.Tensor,
                mean: tuple[float, float, float] = IMAGENET_MEAN,
                std: tuple[float, float, float] = IMAGENET_STD) -> np.ndarray:
    """Reverse Normalize, return (H, W, 3) uint8 RGB."""
    if isinstance(image_chw, torch.Tensor):
        img = image_chw.detach().cpu().numpy()
    else:
        img = np.asarray(image_chw, dtype=np.float32)
    img = np.transpose(img, (1, 2, 0))
    img = img * np.array(std, dtype=np.float32) + np.array(mean, dtype=np.float32)
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


# --- preprocessing --------------------------------------------------------

def preprocess_image_pil(
    image: Image.Image | np.ndarray | str,
    image_size: int = 640,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> tuple[torch.Tensor, dict]:
    """Resize (longest side) + pad top-left to (image_size, image_size) + normalize.

    Works great with PIL. Returns both the input tensor AND the metadata needed
    to map boxes back to the original image.

    Args:
        image: PIL.Image, (H, W, 3) uint8 numpy array, or a path to an image file.
        image_size: side of the square network input.

    Returns:
        tensor: (1, 3, image_size, image_size) float32 tensor.
        meta:   dict with:
            - orig_w, orig_h: size of the original image (pixels).
            - scale:           factor applied during resize (same in x and y).
            - pad_left, pad_top: padding added after resize, in network pixels.
    """
    if isinstance(image, str):
        image = Image.open(image)
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode != "RGB":
        image = image.convert("RGB")

    orig_w, orig_h = image.size
    scale = image_size / float(max(orig_w, orig_h))
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))
    resized = image.resize((new_w, new_h), Image.BILINEAR)

    canvas = Image.new("RGB", (image_size, image_size), (0, 0, 0))
    canvas.paste(resized, (0, 0))   # top-left alignment

    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    arr = (arr - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).contiguous()

    meta = {
        "orig_w": orig_w,
        "orig_h": orig_h,
        "scale": scale,
        "pad_left": 0,
        "pad_top": 0,
    }
    return tensor, meta


# --- detection metric (IoU-based H-mean, ICDAR-style) ---------------------

def _poly_iou(a: np.ndarray, b: np.ndarray) -> float:
    pa, pb = Polygon(a), Polygon(b)
    if not pa.is_valid or not pb.is_valid:
        return 0.0
    inter = pa.intersection(pb).area
    union = pa.area + pb.area - inter
    return float(inter / union) if union > 0 else 0.0


def hmean_metric(
    pred_polys: list[list[np.ndarray]],
    gt_polys: list[list[np.ndarray]],
    iou_thresh: float = 0.5,
) -> dict[str, float]:
    """Compute precision / recall / H-mean per ICDAR convention (greedy match).

    Args:
        pred_polys: per-image list of (N, 2) polygons
        gt_polys:   per-image list of (N, 2) polygons
    """
    tp, fp, fn = 0, 0, 0
    for preds, gts in zip(pred_polys, gt_polys):
        matched_gt = [False] * len(gts)
        for p in preds:
            best_iou, best_j = 0.0, -1
            for j, g in enumerate(gts):
                if matched_gt[j]:
                    continue
                iou = _poly_iou(p, g)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= iou_thresh and best_j >= 0:
                tp += 1
                matched_gt[best_j] = True
            else:
                fp += 1
        fn += matched_gt.count(False)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    hmean = 2 * precision * recall / max(precision + recall, 1e-6)
    return {"precision": precision, "recall": recall, "hmean": hmean,
            "tp": tp, "fp": fp, "fn": fn}


# --- EMA ------------------------------------------------------------------

class ModelEMA:
    """Exponential moving average of model parameters (Polyak averaging)."""

    def __init__(self, model: nn.Module, decay: float = 0.9995):
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1.0 - d)
            else:
                v.copy_(msd[k])
