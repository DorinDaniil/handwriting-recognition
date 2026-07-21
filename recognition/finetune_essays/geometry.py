"""Polygon -> upright rectangle crop (rectify a 4-point line polygon and warp it flat).

Same math the inference detector uses (order the quad, optionally grow it along its own
local axes, warp to an upright w×h image with a small background margin) so the fine-tune
crops match what the pipeline feeds the recognizer at test time.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def to_quad(points) -> np.ndarray:
    """Coerce a polygon to a float (4, 2) array. If it isn't 4 points, fall back to its
    axis-aligned bounding box corners."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(pts) == 4:
        return pts
    x0, y0 = pts[:, 0].min(), pts[:, 1].min()
    x1, y1 = pts[:, 0].max(), pts[:, 1].max()
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


def order_quad(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = pts[:, 0] - pts[:, 1]
    return np.stack([pts[np.argmin(s)], pts[np.argmax(d)],
                     pts[np.argmax(s)], pts[np.argmin(d)]]).astype(np.float32)


def expand_quad(quad: np.ndarray, expand_w: float, expand_h: float) -> np.ndarray:
    """Grow (or shrink, for negative fractions) a quad about its centroid by a fraction of its
    width / height. Scales in the image x/y axes, so it works for arbitrary (slanted, non-
    rectangular) hand-labelled quads without rotating or shearing them — the shape is preserved,
    the box just gets a bit larger/smaller. Negative -> crop tighter than the labelled box."""
    q = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    if expand_w == 0.0 and expand_h == 0.0:
        return q.astype(np.float32)
    c = q.mean(axis=0)
    scale = np.array([1.0 + expand_w, 1.0 + expand_h], dtype=np.float64)
    return (c + (q - c) * scale).astype(np.float32)


def estimate_bg(rgb: np.ndarray) -> tuple[int, int, int]:
    flat = rgb.reshape(-1, 3)
    if len(flat) > 200_000:
        flat = flat[np.random.default_rng(0).choice(len(flat), 200_000, replace=False)]
    return tuple(int(v) for v in np.median(flat, axis=0))


def warp_crop(rgb: np.ndarray, quad: np.ndarray, bg, margin_frac: float = 0.06):
    """Warp the (ordered) quad to an upright crop, padded with a small `bg` margin."""
    q = order_quad(quad)
    w = max(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3]))
    h = max(np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[2] - q[1]))
    w, h = int(round(w)), int(round(h))
    if w < 2 or h < 2:
        return None
    tl, tr, br, bl = q
    data = (tl[0], tl[1], bl[0], bl[1], br[0], br[1], tr[0], tr[1])
    src = Image.fromarray(rgb)
    try:
        line = src.transform((w, h), Image.QUAD, data, resample=Image.BILINEAR, fillcolor=bg)
    except (TypeError, ValueError):
        line = src.transform((w, h), Image.QUAD, data, resample=Image.BILINEAR)
    if margin_frac > 0:
        m = max(1, int(round(h * margin_frac)))
        canvas = Image.new("RGB", (w + 2 * m, h + 2 * m), bg)
        canvas.paste(line, (m, m))
        line = canvas
    return np.asarray(line)
