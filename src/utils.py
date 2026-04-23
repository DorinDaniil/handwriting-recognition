"""Small utilities: seeding, visualization, detection H-mean metric, EMA."""
from __future__ import annotations

import colorsys
import copy
import logging
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)


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
    image: np.ndarray,
    polygons: list[np.ndarray],
    scores: list[float] | None = None,
    thickness: int = 2,
) -> np.ndarray:
    """Draw polygons on a (H, W, 3) uint8 RGB image. Returns a new image."""
    out = image.copy()
    n = len(polygons)
    for i, poly in enumerate(polygons):
        pts = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
        color = _color_for(i, n)
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=thickness)
        if scores is not None:
            x, y = int(poly[0][0]), int(poly[0][1])
            cv2.putText(out, f"{scores[i]:.2f}", (x, max(0, y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def denormalize(image_chw: torch.Tensor,
                mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
                std: tuple[float, float, float] = (0.229, 0.224, 0.225)) -> np.ndarray:
    """Reverse Normalize, return (H, W, 3) uint8 RGB."""
    img = image_chw.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = img * np.array(std, dtype=np.float32) + np.array(mean, dtype=np.float32)
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


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
