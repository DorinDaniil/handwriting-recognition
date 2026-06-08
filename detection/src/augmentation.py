"""Augmentation pipeline for DBNet++ training.

Deliberately simple and polygon-safe:
    - discrete geometry only (90/180/270 rotations + horizontal/vertical flips).
      Continuous affine rotations, random scale+crop, elastic and perspective
      are intentionally avoided — they tend to drift pseudo-labels even when
      keypoints technically follow.
    - mild photometric: color jitter, light blur, light noise.

Tiers:
    - none         : resize + pad + normalize (validation / inference).
    - standard     : resize + pad + discrete rotate/flip + mild color/blur/noise.
    - handwriting  : alias of standard for now; kept separate so you can
                     strengthen it later without breaking configs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import cv2
import numpy as np

import albumentations as A

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class AugConfig:
    tier: str = "standard"
    image_size: int = 640
    # geometry — discrete only
    p_rot90: float = 0.75
    p_hflip: float = 0.5
    p_vflip: float = 0.5
    # photometric — kept mild
    p_color_jitter: float = 0.4
    p_blur: float = 0.15
    p_noise: float = 0.15
    # tiny vertex jitter (0 = off) — can absorb a bit of teacher noise
    box_jitter_px: float = 0.0
    mean: tuple[float, float, float] = field(default_factory=lambda: IMAGENET_MEAN)
    std: tuple[float, float, float] = field(default_factory=lambda: IMAGENET_STD)


# --- pipeline factory -----------------------------------------------------

def _keypoint_params() -> A.KeypointParams:
    # keypoints outside the frame are kept so polygons stay consistent;
    # we clip them back inside at the dataset level.
    return A.KeypointParams(format="xy", remove_invisible=False, angle_in_degrees=True)


def _pad_if_needed(size: int) -> A.BasicTransform:
    try:
        return A.PadIfNeeded(min_height=size, min_width=size,
                             border_mode=cv2.BORDER_CONSTANT, fill=0, p=1.0)
    except TypeError:
        return A.PadIfNeeded(min_height=size, min_width=size,
                             border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0)


def _geometric(cfg: AugConfig) -> list[A.BasicTransform]:
    """Resize + pad to square, then discrete rotations and flips only."""
    return [
        A.LongestMaxSize(max_size=cfg.image_size, p=1.0),
        _pad_if_needed(cfg.image_size),
        A.RandomRotate90(p=cfg.p_rot90),
        A.HorizontalFlip(p=cfg.p_hflip),
        A.VerticalFlip(p=cfg.p_vflip),
    ]


def _photometric(cfg: AugConfig) -> list[A.BasicTransform]:
    blur_group = A.OneOf(
        [
            A.GaussianBlur(blur_limit=(3, 5)),
            A.MotionBlur(blur_limit=5),
        ],
        p=cfg.p_blur,
    )
    # GaussNoise kwargs renamed across albumentations versions
    try:
        gn = A.GaussNoise(std_range=(0.01, 0.04))
    except TypeError:
        gn = A.GaussNoise(var_limit=(5.0, 20.0))
    noise_group = A.OneOf([gn, A.ISONoise()], p=cfg.p_noise)
    return [
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02,
                      p=cfg.p_color_jitter),
        A.ToGray(p=0.05),
        blur_group,
        noise_group,
    ]


def _normalize(cfg: AugConfig) -> list[A.BasicTransform]:
    return [A.Normalize(mean=cfg.mean, std=cfg.std, max_pixel_value=255.0)]


def build_transform(cfg: AugConfig, train: bool) -> A.Compose:
    ops: list[A.BasicTransform]
    if not train or cfg.tier == "none":
        ops = [
            A.LongestMaxSize(max_size=cfg.image_size, p=1.0),
            _pad_if_needed(cfg.image_size),
            *_normalize(cfg),
        ]
    elif cfg.tier in ("standard", "handwriting"):
        ops = [
            *_geometric(cfg),
            *_photometric(cfg),
            *_normalize(cfg),
        ]
    else:
        raise ValueError(f"Unknown aug tier: {cfg.tier}")

    return A.Compose(ops, keypoint_params=_keypoint_params())


# --- top-level Augmenter --------------------------------------------------

class Augmenter:
    """
    Thin wrapper that (1) flattens polygons to keypoints, (2) applies albumentations,
    (3) re-forms polygons, clips out-of-frame vertices, drops degenerate polys,
    and (4) optionally jitters remaining vertices by a few pixels.
    """

    def __init__(self, cfg: AugConfig, train: bool):
        self.cfg = cfg
        self.train = train
        self.pipeline = build_transform(cfg, train)

    def __call__(
        self,
        image: np.ndarray,
        polygons: Sequence[np.ndarray],
    ) -> tuple[np.ndarray, list[np.ndarray], list[bool]]:
        """Transform an image + its polygons.

        Returns:
            image_t:    (H, W, 3) float32 in normalized range
            polys_t:    list of (N_i, 2) float32 polygons in transformed pixels
            keep:       list[bool] per input polygon — False if dropped
        """
        polygons = [np.asarray(p, dtype=np.float32).reshape(-1, 2) for p in polygons]
        sizes = [len(p) for p in polygons]
        flat_kp = np.concatenate(polygons, axis=0) if sizes else np.zeros((0, 2), np.float32)

        out = self.pipeline(image=image, keypoints=flat_kp.tolist())
        img_t = out["image"]
        kp_t = np.asarray(out["keypoints"], dtype=np.float32).reshape(-1, 2) if out["keypoints"] else np.zeros((0, 2), np.float32)

        h, w = img_t.shape[:2]
        polys_t: list[np.ndarray] = []
        keep: list[bool] = []
        cursor = 0
        for n in sizes:
            pts = kp_t[cursor:cursor + n]
            cursor += n
            if len(pts) < 3:
                keep.append(False)
                continue
            if self.train and self.cfg.box_jitter_px > 0:
                jitter = np.random.uniform(-self.cfg.box_jitter_px,
                                           self.cfg.box_jitter_px, size=pts.shape)
                pts = pts + jitter.astype(np.float32)
            pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
            # drop if degenerate after clipping
            x_range = pts[:, 0].max() - pts[:, 0].min()
            y_range = pts[:, 1].max() - pts[:, 1].min()
            if min(x_range, y_range) < 2:
                keep.append(False)
                continue
            polys_t.append(pts)
            keep.append(True)
        return img_t, polys_t, keep
