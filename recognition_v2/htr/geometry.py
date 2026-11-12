"""Line-box geometry: the one crop used by training, evaluation and inference.

A line is a quad on the page. It is grown by a frame given in units of its own height and
warped upright, so the crop always carries the same amount of real page around the text,
whatever the size of the box.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class Frame:
    """Margins around a line box, in box heights."""
    left: float
    right: float
    top: float
    bottom: float

    @classmethod
    def uniform(cls, x: float, y: float) -> "Frame":
        return cls(x, x, y, y)

    @classmethod
    def random(cls, rng, x, y) -> "Frame":
        return cls(rng.uniform(*x), rng.uniform(*x), rng.uniform(*y), rng.uniform(*y))


def order_quad(points) -> np.ndarray:
    """Any polygon -> 4 corners: top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) != 4:
        (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
        pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    s, d = pts.sum(1), pts[:, 0] - pts[:, 1]
    return np.stack([pts[s.argmin()], pts[d.argmax()], pts[s.argmax()], pts[d.argmin()]])


def grow(quad, frame: Frame) -> np.ndarray:
    """Move each side of the quad outward along the box's own axes."""
    tl, tr, br, bl = order_quad(quad)
    along, down = (tr - tl + br - bl) / 2, (bl - tl + br - tr) / 2
    width, height = np.linalg.norm(along), np.linalg.norm(down)
    if width < 1e-6 or height < 1e-6:
        return np.stack([tl, tr, br, bl])
    x, y = along / width * height, down
    return np.stack([tl - frame.left * x - frame.top * y, tr + frame.right * x - frame.top * y,
                     br + frame.right * x + frame.bottom * y, bl - frame.left * x + frame.bottom * y])


def warp(page: Image.Image, quad: np.ndarray, fill) -> Image.Image | None:
    """Rectify an ordered quad into an upright image; pixels outside the page take `fill`."""
    tl, tr, br, bl = quad
    width = round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    height = round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    if width < 2 or height < 2:
        return None
    corners = (*tl, *bl, *br, *tr)
    return page.transform((width, height), Image.QUAD, corners, resample=Image.BILINEAR, fillcolor=fill)


def crop_line(page: Image.Image, quad, frame: Frame, fill) -> Image.Image | None:
    return warp(page, grow(quad, frame), fill)


def paper_colour(image: Image.Image) -> tuple[int, int, int]:
    small = np.asarray(image.convert("RGB").resize((64, 64), Image.BILINEAR)).reshape(-1, 3)
    return tuple(int(v) for v in np.median(small, axis=0))


def _area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _clip(subject: np.ndarray, clipper: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman: the part of `subject` inside the convex polygon `clipper`."""
    if _area(clipper) < 0:
        clipper = clipper[::-1]
    out = [tuple(p) for p in subject]
    for a, b in zip(clipper, np.roll(clipper, -1, axis=0)):
        if not out:
            break
        edge = b - a
        side = [edge[0] * (p[1] - a[1]) - edge[1] * (p[0] - a[0]) for p in out]
        kept = []
        for i, p in enumerate(out):
            q, sp, sq = out[i - 1], side[i], side[i - 1]
            if (sp >= 0) != (sq >= 0):
                t = sq / (sq - sp)
                kept.append((q[0] + t * (p[0] - q[0]), q[1] + t * (p[1] - q[1])))
            if sp >= 0:
                kept.append(p)
        out = kept
    return np.asarray(out, dtype=np.float64).reshape(-1, 2)


def overlap(a, b) -> float:
    inter = _clip(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))
    return abs(_area(inter)) if len(inter) >= 3 else 0.0


def match_detections(labelled, detected, min_overlap=0.6, min_span=0.85, max_span=1.15, max_height=1.6):
    """For every labelled line, the index of the detector box that is this same whole line, or None.

    A box qualifies when the smaller of the two shapes lies mostly inside the other, it spans the
    labelled line along its length (not a fragment, not two lines side by side) and is not much
    taller (not two lines stacked). A box claimed by several labelled lines is given to none.
    """
    labelled = [order_quad(q) for q in labelled]
    detected = [order_quad(q) for q in detected]
    choice = []
    for line in labelled:
        axis = line[1] - line[0]
        axis = axis / max(np.linalg.norm(axis), 1e-6)
        lo, hi = (line @ axis).min(), (line @ axis).max()
        height = np.linalg.norm(line[3] - line[0])
        best, best_inter = None, 0.0
        for j, box in enumerate(detected):
            inter = overlap(line, box)
            if inter <= best_inter or inter < min_overlap * min(abs(_area(line)), abs(_area(box))):
                continue
            a, b = (box @ axis).min(), (box @ axis).max()
            covered = (min(hi, b) - max(lo, a)) / max(hi - lo, 1e-6)
            if covered < min_span or (b - a) > max_span * (hi - lo):
                continue
            if np.linalg.norm(box[3] - box[0]) > max_height * height:
                continue
            best, best_inter = j, inter
        choice.append(best)
    claimed = [j for j in choice if j is not None]
    return [j if j is not None and claimed.count(j) == 1 else None for j in choice]
