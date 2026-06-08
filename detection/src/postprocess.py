"""DBNet++ post-processing: probability map -> polygons. Pure numpy/shapely.

Pipeline:
    1. Binarize:       mask = prob > cfg.thresh
    2. Connected components via scipy.ndimage.label + find_objects
    3. For each component (work scoped to its bbox — fast even with thousands):
        a. Bbox / area pre-filter
        b. Mean prob check (>= cfg.box_thresh)
        c. Boundary pixels via 1-pixel erosion (cheap and gives ~perimeter points,
           instead of all interior pixels — orders of magnitude faster than
           MultiPoint over the full region)
        d. Min rotated rect on those boundary points (shapely)
        e. Unclip by cfg.unclip_ratio via pyclipper (Vatti offset)
        f. Final min rotated rect on the expanded polygon
    4. Undo preprocess transform (pad + scale) to original image coords.

No OpenCV anywhere.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyclipper
from scipy.ndimage import binary_erosion, find_objects, label as nd_label
from shapely.geometry import MultiPoint, Polygon


@dataclass
class PostprocessConfig:
    thresh: float = 0.3
    box_thresh: float = 0.5
    unclip_ratio: float = 1.8
    max_candidates: int = 1000
    min_size: int = 3


# --- helpers --------------------------------------------------------------

def _rect_short_side(rect: np.ndarray) -> float:
    """rect: (4, 2) points of a rotated rectangle in order around the perimeter."""
    d1 = np.linalg.norm(rect[1] - rect[0])
    d2 = np.linalg.norm(rect[2] - rect[1])
    return float(min(d1, d2))


def _min_rotated_rect(points: np.ndarray) -> np.ndarray | None:
    """Return the 4 corner points of the minimum rotated rectangle around `points`."""
    if len(points) < 3:
        return None
    try:
        mrr = MultiPoint(points).minimum_rotated_rectangle
    except Exception:
        return None
    if mrr.is_empty or not hasattr(mrr, "exterior"):
        return None
    coords = np.asarray(mrr.exterior.coords, dtype=np.float32)[:-1]
    if coords.shape != (4, 2):
        return None
    return coords


def _unclip(poly: np.ndarray, unclip_ratio: float) -> np.ndarray | None:
    """Vatti-expand the polygon outward. Returns a new (M, 2) polygon or None."""
    try:
        shapely_poly = Polygon(poly)
        if not shapely_poly.is_valid or shapely_poly.length <= 0:
            return None
        distance = shapely_poly.area * unclip_ratio / shapely_poly.length
    except Exception:
        return None

    offset = pyclipper.PyclipperOffset()
    offset.AddPath([tuple(p) for p in poly],
                   pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = offset.Execute(distance)
    if not expanded:
        return None
    arr = np.asarray(expanded[0], dtype=np.float32)
    if len(arr) < 4:
        return None
    return arr


def _boundary_pixels(region_crop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (xs, ys) of boundary pixels inside `region_crop` (bool 2D).

    Boundary = region minus its 1-pixel erosion. For text-like blobs this
    has O(perimeter) points instead of O(area) — ~20-50x fewer than np.where
    on the full region.
    """
    if region_crop.size == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    eroded = binary_erosion(region_crop, border_value=0)
    boundary = region_crop & ~eroded
    if not boundary.any():
        # very thin / 1-pixel-wide region — fall back to all pixels
        boundary = region_crop
    ys, xs = np.where(boundary)
    return xs, ys


# --- main -----------------------------------------------------------------

def decode_prob_map(
    prob_map: np.ndarray,
    cfg: PostprocessConfig | None = None,
    *,
    scale: float = 1.0,
    pad: tuple[float, float] = (0.0, 0.0),
    original_size: tuple[int, int] | None = None,
) -> tuple[list[np.ndarray], list[float]]:
    """Convert a probability map into a list of rotated-quad polygons.

    Args:
        prob_map:      (H, W) float in [0, 1]. Usually at the network input resolution.
        cfg:           PostprocessConfig. Defaults used if None.
        scale:         preprocessing scale factor (same in x and y). 1.0 means
                       prob_map is already in original-image coords.
        pad:           (pad_left, pad_top) padding added during preprocessing,
                       measured in prob-map pixels.
        original_size: (orig_width, orig_height) — PIL convention. If given,
                       boxes are clipped to it.

    Returns:
        boxes:  list of (4, 2) float32 quads in ORIGINAL image coords.
        scores: list of per-box mean probabilities.
    """
    cfg = cfg or PostprocessConfig()
    pad_x, pad_y = float(pad[0]), float(pad[1])
    inv_scale = 1.0 / max(float(scale), 1e-8)
    min_area = max(1, cfg.min_size * cfg.min_size)

    mask = prob_map > cfg.thresh
    labels, num = nd_label(mask)
    if num == 0:
        return [], []
    slices = find_objects(labels)

    boxes: list[np.ndarray] = []
    scores: list[float] = []

    for idx, sl in enumerate(slices, start=1):
        if len(boxes) >= cfg.max_candidates:
            break
        if sl is None:
            continue

        # quick bbox-area pre-filter — cheap, kills tiny noise components
        bbox_h = sl[0].stop - sl[0].start
        bbox_w = sl[1].stop - sl[1].start
        if bbox_h * bbox_w < min_area:
            continue

        # all subsequent work is on the bbox crop only
        crop_region = labels[sl] == idx
        area = int(crop_region.sum())
        if area < min_area:
            continue

        crop_prob = prob_map[sl]
        s = float(crop_prob[crop_region].mean())
        if s < cfg.box_thresh:
            continue

        # boundary points only (perimeter, not area) — huge speedup
        xs_local, ys_local = _boundary_pixels(crop_region)
        if len(xs_local) < 3:
            continue

        y0, x0 = sl[0].start, sl[1].start
        pts = np.stack([xs_local + x0, ys_local + y0], axis=1).astype(np.float32)

        rect0 = _min_rotated_rect(pts)
        if rect0 is None or _rect_short_side(rect0) < cfg.min_size:
            continue

        expanded = _unclip(rect0, cfg.unclip_ratio)
        if expanded is None:
            continue

        rect1 = _min_rotated_rect(expanded)
        if rect1 is None or _rect_short_side(rect1) < cfg.min_size + 2:
            continue

        # prob-map coords -> original image coords
        quad = rect1.copy()
        quad[:, 0] = (quad[:, 0] - pad_x) * inv_scale
        quad[:, 1] = (quad[:, 1] - pad_y) * inv_scale

        if original_size is not None:
            ow, oh = original_size
            quad[:, 0] = np.clip(quad[:, 0], 0, ow - 1)
            quad[:, 1] = np.clip(quad[:, 1], 0, oh - 1)

        boxes.append(quad.astype(np.float32))
        scores.append(s)

    return boxes, scores
