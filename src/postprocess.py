"""DBNet++ post-processing: probability map -> polygons. Pure numpy/shapely.

Pipeline:
    1. Binarize:       mask = prob > cfg.thresh
    2. Connected components via scipy.ndimage.label
    3. For each component:
        a. Mean prob check (>= cfg.box_thresh)
        b. Fit minimum rotated rectangle (shapely)
        c. Check short side (>= cfg.min_size)
        d. Unclip by cfg.unclip_ratio via pyclipper (Vatti offset)
        e. Fit min rotated rect to the expanded polygon
    4. Undo preprocess transform (pad + scale) to go back to original image coords.

No OpenCV anywhere.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyclipper
from scipy.ndimage import label as nd_label
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
    """rect: (4, 2) points of a rotated rectangle, in order around the perimeter."""
    d1 = np.linalg.norm(rect[1] - rect[0])
    d2 = np.linalg.norm(rect[2] - rect[1])
    return float(min(d1, d2))


def _min_rotated_rect(points: np.ndarray) -> np.ndarray | None:
    """Return the 4 corner points of the minimum rotated rectangle around `points`."""
    if len(points) < 2:
        return None
    try:
        mrr = MultiPoint(points).minimum_rotated_rectangle
    except Exception:
        return None
    if mrr.is_empty or not hasattr(mrr, "exterior"):
        return None
    coords = np.asarray(mrr.exterior.coords, dtype=np.float32)[:-1]  # drop repeated last
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
        scale:         preprocessing scale factor (same in x and y). Set to the value
                       you used to resize the original image into the padded canvas.
                       Passing 1.0 means `prob_map` already sits in original coords.
        pad:           (pad_left, pad_top) in prob-map pixels. The padding you added
                       when building the square network input.
        original_size: (orig_width, orig_height) in the ORIGINAL image. Same order as
                       PIL's `img.size`. If given, boxes are clipped to it.

    Returns:
        boxes:  list of (4, 2) float32 quads in ORIGINAL image coords.
        scores: list of per-box mean probabilities.
    """
    cfg = cfg or PostprocessConfig()
    pad_x, pad_y = float(pad[0]), float(pad[1])
    inv_scale = 1.0 / max(float(scale), 1e-8)

    mask = prob_map > cfg.thresh
    labels, num = nd_label(mask)

    boxes: list[np.ndarray] = []
    scores: list[float] = []

    for i in range(1, num + 1):
        if len(boxes) >= cfg.max_candidates:
            break

        region = labels == i
        area = int(region.sum())
        if area < cfg.min_size * cfg.min_size:
            continue

        # mean probability inside the component
        s = float(prob_map[region].mean())
        if s < cfg.box_thresh:
            continue

        # fit a rotated rectangle to all foreground pixels of this component
        ys, xs = np.where(region)
        pts = np.stack([xs, ys], axis=1).astype(np.float32)     # (N, 2) in (x, y)
        rect0 = _min_rotated_rect(pts)
        if rect0 is None or _rect_short_side(rect0) < cfg.min_size:
            continue

        # grow the rectangle by the DB unclip distance
        expanded = _unclip(rect0, cfg.unclip_ratio)
        if expanded is None:
            continue

        # final rotated rectangle around the expanded polygon
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
