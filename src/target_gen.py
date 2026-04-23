"""DBNet++ target generation: shrink map, threshold map, and masks.

Follows the label-generation recipe from "Real-Time Scene Text Detection with
Differentiable Binarization and Adaptive Scale Fusion" (arXiv:2202.10304).

Per polygon:
    - shrink offset  D = A * (1 - r^2) / L     (r = shrink_ratio)
    - positive region = Vatti-shrunk polygon  -> probability map = 1
    - border region   = ring of width D around original poly -> threshold map in [t_min, t_max]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon


@dataclass
class TargetConfig:
    shrink_ratio: float = 0.4
    thresh_min: float = 0.3
    thresh_max: float = 0.7
    min_text_size: int = 4


def _polygon_area_perimeter(poly: np.ndarray) -> tuple[float, float]:
    p = Polygon(poly)
    return float(abs(p.area)), float(p.length)


def _shrink_polygon(poly: np.ndarray, ratio: float) -> np.ndarray | None:
    """Vatti-clip a polygon inwards. Returns shrunk poly or None if degenerate."""
    area, perimeter = _polygon_area_perimeter(poly)
    if perimeter < 1e-6:
        return None
    distance = area * (1.0 - ratio ** 2) / perimeter
    subj = [tuple(p) for p in poly]
    pco = pyclipper.PyclipperOffset()
    pco.AddPath(subj, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    shrunk = pco.Execute(-distance)
    if not shrunk:
        return None
    shrunk = np.array(shrunk[0])
    if shrunk.size == 0 or len(shrunk) < 3:
        return None
    return shrunk


def _expand_polygon(poly: np.ndarray, ratio: float) -> tuple[np.ndarray | None, float]:
    """Expand polygon outwards by the same distance the shrink uses."""
    area, perimeter = _polygon_area_perimeter(poly)
    if perimeter < 1e-6:
        return None, 0.0
    distance = area * (1.0 - ratio ** 2) / perimeter
    subj = [tuple(p) for p in poly]
    pco = pyclipper.PyclipperOffset()
    pco.AddPath(subj, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = pco.Execute(distance)
    if not expanded:
        return None, distance
    expanded = np.array(expanded[0])
    if expanded.size == 0 or len(expanded) < 3:
        return None, distance
    return expanded, distance


def _point_to_segment_dist(xs: np.ndarray, ys: np.ndarray,
                           ax: float, ay: float, bx: float, by: float) -> np.ndarray:
    """Per-pixel distance to segment AB. xs, ys: arrays of same shape."""
    abx, aby = bx - ax, by - ay
    apx, apy = xs - ax, ys - ay
    seg_len_sq = abx * abx + aby * aby
    if seg_len_sq < 1e-6:
        return np.sqrt(apx * apx + apy * apy)
    t = (apx * abx + apy * aby) / seg_len_sq
    t = np.clip(t, 0.0, 1.0)
    proj_x = ax + t * abx
    proj_y = ay + t * aby
    dx = xs - proj_x
    dy = ys - proj_y
    return np.sqrt(dx * dx + dy * dy)


def _fill_threshold_map(canvas: np.ndarray, mask: np.ndarray,
                        expanded_poly: np.ndarray, orig_poly: np.ndarray,
                        distance: float) -> None:
    """
    Inside the bounding box of the expanded polygon, compute per-pixel distance
    to the original polygon's boundary, normalize by `distance`, invert, clip.
    Writes the per-pixel max of the previous value and the new contribution.
    """
    h, w = canvas.shape
    xmin = int(max(0, np.floor(expanded_poly[:, 0].min())))
    xmax = int(min(w - 1, np.ceil(expanded_poly[:, 0].max())))
    ymin = int(max(0, np.floor(expanded_poly[:, 1].min())))
    ymax = int(min(h - 1, np.ceil(expanded_poly[:, 1].max())))
    if xmax < xmin or ymax < ymin:
        return

    xs, ys = np.meshgrid(
        np.arange(xmin, xmax + 1, dtype=np.float32),
        np.arange(ymin, ymax + 1, dtype=np.float32),
    )

    n = len(orig_poly)
    dist_map = np.full(xs.shape, np.inf, dtype=np.float32)
    for i in range(n):
        ax, ay = orig_poly[i]
        bx, by = orig_poly[(i + 1) % n]
        d = _point_to_segment_dist(xs, ys, ax, ay, bx, by)
        dist_map = np.minimum(dist_map, d)

    # inside ring: 0 on boundary, 1 at shrink-distance away
    val = dist_map / max(distance, 1e-6)
    val = np.clip(val, 0.0, 1.0)
    val = 1.0 - val
    # apply only inside expanded polygon mask
    region_mask = mask[ymin:ymax + 1, xmin:xmax + 1] > 0
    dst = canvas[ymin:ymax + 1, xmin:xmax + 1]
    dst[region_mask] = np.maximum(dst[region_mask], val[region_mask])
    canvas[ymin:ymax + 1, xmin:xmax + 1] = dst


def generate_targets(
    image_shape: tuple[int, int],
    polygons: Sequence[np.ndarray],
    ignore_flags: Sequence[bool] | None = None,
    cfg: TargetConfig | None = None,
) -> dict[str, np.ndarray]:
    """
    Build DBNet++ training targets.

    Args:
        image_shape: (H, W)
        polygons:    list of (N, 2) float arrays, image-pixel coordinates.
        ignore_flags: per-polygon flag; True = don't supervise (but mask out loss).

    Returns dict:
        prob_map    : (H, W) float32  — 1 inside shrunk polygon
        prob_mask   : (H, W) float32  — 1 = pixel contributes to prob/binary loss
        thresh_map  : (H, W) float32  — values in [t_min, t_max] on border ring, 0 elsewhere
        thresh_mask : (H, W) float32  — 1 on border ring (where thresh loss applies)
    """
    cfg = cfg or TargetConfig()
    H, W = image_shape
    if ignore_flags is None:
        ignore_flags = [False] * len(polygons)

    prob_map = np.zeros((H, W), dtype=np.float32)
    prob_mask = np.ones((H, W), dtype=np.float32)
    thresh_map = np.zeros((H, W), dtype=np.float32)
    thresh_mask = np.zeros((H, W), dtype=np.float32)

    for poly, ignore in zip(polygons, ignore_flags):
        poly = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
        if len(poly) < 3:
            continue

        # clip coords to image
        poly[:, 0] = np.clip(poly[:, 0], 0, W - 1)
        poly[:, 1] = np.clip(poly[:, 1], 0, H - 1)

        w_box = poly[:, 0].max() - poly[:, 0].min()
        h_box = poly[:, 1].max() - poly[:, 1].min()
        if min(w_box, h_box) < cfg.min_text_size:
            ignore = True

        if ignore:
            cv2.fillPoly(prob_mask, [poly.astype(np.int32)], 0.0)
            continue

        # probability map = shrunk polygon
        shrunk = _shrink_polygon(poly, cfg.shrink_ratio)
        if shrunk is None:
            # too small to shrink — ignore in loss
            cv2.fillPoly(prob_mask, [poly.astype(np.int32)], 0.0)
            continue
        cv2.fillPoly(prob_map, [shrunk.astype(np.int32)], 1.0)

        # threshold map = ring between expanded and shrunk polygon
        expanded, dist = _expand_polygon(poly, cfg.shrink_ratio)
        if expanded is None or dist < 1e-6:
            continue
        # ring mask = inside expanded minus inside shrunk
        ring = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(ring, [expanded.astype(np.int32)], 1)
        cv2.fillPoly(ring, [shrunk.astype(np.int32)], 0)
        thresh_mask = np.maximum(thresh_mask, ring.astype(np.float32))

        _fill_threshold_map(thresh_map, ring, expanded, poly, dist)

    # rescale threshold map from [0, 1] to [t_min, t_max]
    thresh_map = thresh_map * (cfg.thresh_max - cfg.thresh_min) + cfg.thresh_min
    thresh_map = thresh_map * thresh_mask  # zero outside ring

    return {
        "prob_map": prob_map,
        "prob_mask": prob_mask,
        "thresh_map": thresh_map,
        "thresh_mask": thresh_mask,
    }
