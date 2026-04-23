"""DBNet++ post-processing: probability map -> polygons.

Steps:
    1. Binarize: mask = prob > thresh
    2. Find contours (CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE).
    3. For each contour:
        a. Filter by mean probability >= box_thresh.
        b. Filter by min_size.
        c. Unclip (Vatti-expand) by unclip_ratio.
        d. Fit minAreaRect -> 4-point polygon.
    4. Rescale to original image size.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon


@dataclass
class PostprocessConfig:
    thresh: float = 0.3
    box_thresh: float = 0.5
    unclip_ratio: float = 1.8
    max_candidates: int = 1000
    min_size: int = 3


def _unclip(poly: np.ndarray, unclip_ratio: float) -> np.ndarray | None:
    polygon = Polygon(poly)
    distance = polygon.area * unclip_ratio / max(polygon.length, 1e-6)
    offset = pyclipper.PyclipperOffset()
    offset.AddPath([tuple(p) for p in poly],
                   pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = offset.Execute(distance)
    if not expanded:
        return None
    expanded = np.array(expanded[0])
    if len(expanded) < 4:
        return None
    return expanded


def _mean_score(prob_map: np.ndarray, contour: np.ndarray) -> float:
    h, w = prob_map.shape
    xmin = int(max(0, np.floor(contour[:, 0].min())))
    xmax = int(min(w - 1, np.ceil(contour[:, 0].max())))
    ymin = int(max(0, np.floor(contour[:, 1].min())))
    ymax = int(min(h - 1, np.ceil(contour[:, 1].max())))
    if xmax <= xmin or ymax <= ymin:
        return 0.0
    crop = prob_map[ymin:ymax + 1, xmin:xmax + 1]
    mask = np.zeros_like(crop, dtype=np.uint8)
    shifted = contour.copy()
    shifted[:, 0] -= xmin
    shifted[:, 1] -= ymin
    cv2.fillPoly(mask, [shifted.astype(np.int32)], 1)
    if mask.sum() == 0:
        return 0.0
    return float(cv2.mean(crop, mask)[0])


def _min_rect_quad(contour: np.ndarray) -> tuple[np.ndarray, float]:
    rect = cv2.minAreaRect(contour.astype(np.float32))
    box = cv2.boxPoints(rect)
    (_, _), (w, h), _ = rect
    return box, min(w, h)


def decode_prob_map(
    prob_map: np.ndarray,
    original_size: tuple[int, int],
    cfg: PostprocessConfig | None = None,
) -> tuple[list[np.ndarray], list[float]]:
    """
    Args:
        prob_map: (H, W) float32 probability in [0, 1] at network resolution.
        original_size: (orig_h, orig_w) image size to rescale polygons back to.
    Returns:
        boxes:  list of (4, 2) float32 polygons in original image coords.
        scores: list of per-box mean prob.
    """
    cfg = cfg or PostprocessConfig()
    net_h, net_w = prob_map.shape
    orig_h, orig_w = original_size

    mask = (prob_map > cfg.thresh).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[: cfg.max_candidates]

    boxes: list[np.ndarray] = []
    scores: list[float] = []
    for c in contours:
        if len(c) < 3:
            continue
        c = c.reshape(-1, 2)

        # score filter on the raw contour
        s = _mean_score(prob_map, c)
        if s < cfg.box_thresh:
            continue

        # minAreaRect on the raw contour (cheap, gives initial size check)
        _, short_side = _min_rect_quad(c)
        if short_side < cfg.min_size:
            continue

        # unclip and fit quad to the expanded polygon
        expanded = _unclip(c, cfg.unclip_ratio)
        if expanded is None:
            continue
        quad, short_side2 = _min_rect_quad(expanded.reshape(-1, 2))
        if short_side2 < cfg.min_size + 2:
            continue

        # rescale to original resolution
        quad = quad.astype(np.float32)
        quad[:, 0] = np.clip(quad[:, 0] / net_w * orig_w, 0, orig_w - 1)
        quad[:, 1] = np.clip(quad[:, 1] / net_h * orig_h, 0, orig_h - 1)

        boxes.append(quad)
        scores.append(s)

    return boxes, scores
