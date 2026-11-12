"""From the two maps of the page model to the pages of a photo.

`paper` marks every pixel that belongs to a page. `core` marks the inside of each page with its
border peeled off, so two pages that touch on the paper map (a spread) still have separate cores.
Every blob of the core map is one page; each paper pixel then goes to the nearest core, the
outline of the blob gives four corners, and the page is cut out upright.

    labels, n = split(paper, core)                 # 0 background, 1..n pages
    quads = [corners(labels == k) for k in ...]    # top-left, top-right, bottom-right, bottom-left
    page = cut(photo, quad)                        # page.image goes to the line detector
    page.to_photo(points)                          # detector boxes back to photo coordinates
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import ConvexHull

MIN_SHARE = 0.015       # a page core covers at least this share of the map
REACH = 0.32            # paper further than this (in sqrt of the core area) from a core is not its page


def split(paper: np.ndarray, core: np.ndarray):
    """Probability maps [h, w] -> (label map [h, w]: 0 background, 1..n pages, n)."""
    blobs, found = ndimage.label(core > 0.5)
    sizes = ndimage.sum_labels(np.ones_like(blobs), blobs, np.arange(1, found + 1)) if found else []
    kept = [k + 1 for k, size in enumerate(sizes) if size >= MIN_SHARE * core.size]
    if not kept:
        return np.zeros(core.shape, np.int32), 0

    cores = np.zeros(core.shape, np.int32)
    reach = np.zeros(len(kept) + 1)
    for new, old in enumerate(kept, 1):
        cores[blobs == old] = new
        reach[new] = REACH * np.sqrt(sizes[old - 1])

    distance, (rows, cols) = ndimage.distance_transform_edt(cores == 0, return_indices=True)
    nearest = cores[rows, cols]
    labels = np.where((paper > 0.5) & (distance <= reach[nearest]), nearest, 0)
    labels[cores > 0] = cores[cores > 0]
    return labels.astype(np.int32), len(kept)


def order_corners(points) -> np.ndarray:
    """4 points -> top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(points, np.float64).reshape(4, 2)
    s, d = pts.sum(1), pts[:, 0] - pts[:, 1]
    return pts[[np.argmin(s), np.argmax(d), np.argmax(s), np.argmin(d)]]


def corners(mask: np.ndarray) -> np.ndarray:
    """Four corners (x, y) of a page blob. The blob is wrapped in its narrowest rectangle and the
    corners are the outline points that reach furthest into the rectangle's corners, so a page
    seen at an angle keeps its real, non-rectangular shape."""
    rows, cols = np.nonzero(mask & ~ndimage.binary_erosion(mask))
    pts = np.stack([cols, rows], axis=1).astype(np.float64)
    if np.ptp(cols) < 2 or np.ptp(rows) < 2:
        x0, y0, x1, y1 = cols.min(), rows.min(), cols.max(), rows.max()
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float64)

    hull = pts[ConvexHull(pts).vertices]
    edges = np.diff(np.vstack([hull, hull[:1]]), axis=0)
    best, best_area = None, np.inf
    for angle in np.unique(np.round(np.arctan2(edges[:, 1], edges[:, 0]) % (np.pi / 2), 3)):
        c, s = np.cos(angle), np.sin(angle)
        u, v = hull[:, 0] * c + hull[:, 1] * s, -hull[:, 0] * s + hull[:, 1] * c
        area = np.ptp(u) * np.ptp(v)
        if area < best_area:
            best, best_area = (u, v), area
    u, v = best
    return order_corners(hull[[np.argmin(u + v), np.argmax(u - v), np.argmax(u + v), np.argmin(u - v)]])


@dataclass
class Page:
    image: Image.Image          # the page, upright
    quad: np.ndarray            # where the corners of `image` are in the photo: TL, TR, BR, BL

    def to_photo(self, points) -> np.ndarray:
        """Points (x, y) of `image` -> the same points in the photo."""
        p = np.asarray(points, np.float64)
        u, v = p[..., :1] / self.image.width, p[..., 1:] / self.image.height
        tl, tr, br, bl = self.quad
        return (1 - u) * (1 - v) * tl + u * (1 - v) * tr + u * v * br + (1 - u) * v * bl


def cut(photo: Image.Image, quad, margin: float = 0.03) -> Page:
    """Warp the quad (grown by `margin` of its size, so nothing at the paper's edge is lost) into
    an upright image. Outside the photo the crop is black."""
    quad = np.asarray(quad, np.float64)
    quad = quad.mean(0) + (quad - quad.mean(0)) * (1 + 2 * margin)
    tl, tr, br, bl = quad
    width = round((np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2)
    height = round((np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2)
    image = photo.convert("RGB").transform((max(width, 2), max(height, 2)), Image.QUAD,
                                           (*tl, *bl, *br, *tr), resample=Image.BILINEAR)
    return Page(image, quad)


def reading_order(quads: list) -> list:
    """Pages of a spread go left to right; sheets lying one above another go top to bottom."""
    if len(quads) < 2:
        return list(quads)
    centres = np.array([q.mean(0) for q in quads])
    width = np.mean([np.linalg.norm(q[1] - q[0]) for q in quads])
    axis = 0 if np.ptp(centres[:, 0]) > 0.5 * width else 1
    return [quads[i] for i in np.argsort(centres[:, axis])]


def find(paper: np.ndarray, core: np.ndarray, scale: float, photo_size: tuple) -> list:
    """Maps of the model -> page quads in photo coordinates, in reading order.
    `scale` is map pixels per photo pixel. No page found -> the whole photo is the page."""
    labels, n = split(paper, core)
    quads = [(corners(labels == k) + 0.5) / scale for k in range(1, n + 1)]
    if not quads:
        w, h = photo_size
        quads = [np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64)]
    return reading_order(quads)
