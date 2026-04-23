"""Augmentation pipeline for DBNet++ training on handwriting.

Polygons are transformed as keypoints (flattened vertices) to keep geometry
consistent with the image. Two tiers:

    - standard     : DBNet-style basic geometric + photometric.
    - handwriting  : adds perspective, elastic, shadow, paper-like textures,
                     ink-fade, coarse blurs — useful when pages vary in
                     lighting, scanning quality, and pen pressure.

A 'none' tier only resizes and normalizes (validation / inference).
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
    tier: str = "handwriting"
    image_size: int = 640
    rotate_deg: float = 10.0
    scale_range: tuple[float, float] = (0.6, 1.4)
    box_jitter_px: float = 2.0
    p_elastic: float = 0.2
    p_perspective: float = 0.3
    p_shadow: float = 0.2
    p_paper_texture: float = 0.15
    p_ink_fade: float = 0.15
    p_blur: float = 0.3
    p_noise: float = 0.3
    p_color_jitter: float = 0.5
    mean: tuple[float, float, float] = field(default_factory=lambda: IMAGENET_MEAN)
    std: tuple[float, float, float] = field(default_factory=lambda: IMAGENET_STD)


# --- custom transforms ----------------------------------------------------

class PaperTexture(A.ImageOnlyTransform):
    """Multiply a low-frequency noise map to mimic paper grain / uneven lighting."""

    def __init__(self, strength: float = 0.15, p: float = 0.15):
        super().__init__(p=p)
        self.strength = strength

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        h, w = img.shape[:2]
        low = np.random.rand(max(h // 32, 4), max(w // 32, 4)).astype(np.float32)
        low = cv2.GaussianBlur(low, (0, 0), sigmaX=3)
        low = cv2.resize(low, (w, h), interpolation=cv2.INTER_LINEAR)
        low = 1.0 + self.strength * (2 * low - 1)
        out = img.astype(np.float32) * low[..., None]
        return np.clip(out, 0, 255).astype(img.dtype)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("strength",)


class InkFade(A.ImageOnlyTransform):
    """Increase brightness in a smooth region — simulates faded/thin ink strokes."""

    def __init__(self, p: float = 0.15):
        super().__init__(p=p)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        h, w = img.shape[:2]
        cx = np.random.randint(0, w)
        cy = np.random.randint(0, h)
        radius = np.random.randint(min(h, w) // 6, min(h, w) // 2)
        yy, xx = np.ogrid[:h, :w]
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        mask = np.clip(1.0 - d / max(radius, 1), 0, 1).astype(np.float32)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=radius / 3 + 1)
        out = img.astype(np.float32)
        lift = 60.0 * mask[..., None]
        out = np.clip(out + lift, 0, 255)
        return out.astype(img.dtype)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ()


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


def _affine(cfg: AugConfig) -> A.BasicTransform:
    try:
        return A.Affine(
            rotate=(-cfg.rotate_deg, cfg.rotate_deg),
            scale=cfg.scale_range,
            translate_percent=(0.0, 0.05),
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            p=0.9,
        )
    except TypeError:
        return A.Affine(
            rotate=(-cfg.rotate_deg, cfg.rotate_deg),
            scale=cfg.scale_range,
            translate_percent=(0.0, 0.05),
            mode=cv2.BORDER_CONSTANT,
            cval=0,
            p=0.9,
        )


def _random_crop(size: int) -> A.BasicTransform:
    try:
        return A.RandomCrop(height=size, width=size, pad_if_needed=True, p=1.0)
    except TypeError:
        return A.RandomCrop(height=size, width=size, p=1.0)


def _base_geometric(cfg: AugConfig) -> list[A.BasicTransform]:
    size = cfg.image_size
    # Keep aspect, then pad to square, then random crop of size×size.
    return [
        A.LongestMaxSize(max_size=int(size * cfg.scale_range[1]), p=1.0),
        _pad_if_needed(size),
        _affine(cfg),
        _random_crop(size),
    ]


def _photometric(cfg: AugConfig) -> list[A.BasicTransform]:
    blur_group = A.OneOf(
        [
            A.GaussianBlur(blur_limit=(3, 7)),
            A.MotionBlur(blur_limit=7),
            A.MedianBlur(blur_limit=5),
        ],
        p=cfg.p_blur,
    )
    # GaussNoise / ImageCompression kwargs renamed across albumentations versions
    try:
        gn = A.GaussNoise(std_range=(0.02, 0.1))
    except TypeError:
        gn = A.GaussNoise(var_limit=(10.0, 50.0))
    try:
        ic = A.ImageCompression(quality_range=(50, 95))
    except TypeError:
        ic = A.ImageCompression(quality_lower=50, quality_upper=95)
    noise_group = A.OneOf([gn, A.ISONoise(), ic], p=cfg.p_noise)
    return [
        A.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05,
                      p=cfg.p_color_jitter),
        A.ToGray(p=0.05),
        blur_group,
        noise_group,
    ]


def _shadow() -> A.BasicTransform:
    """Build a RandomShadow compatible with both old and new albumentations APIs."""
    if not hasattr(A, "RandomShadow"):
        return A.NoOp()
    try:
        return A.RandomShadow(
            shadow_roi=(0.0, 0.0, 1.0, 1.0),
            num_shadows_limit=(1, 2),
            shadow_dimension=5,
            p=1.0,
        )
    except TypeError:
        return A.RandomShadow(
            shadow_roi=(0.0, 0.0, 1.0, 1.0),
            num_shadows_lower=1,
            num_shadows_upper=2,
            shadow_dimension=5,
            p=1.0,
        )


def _perspective(p: float) -> A.BasicTransform:
    """A.Perspective args renamed between versions; fall back gracefully."""
    try:
        return A.Perspective(scale=(0.02, 0.06), pad_mode=cv2.BORDER_CONSTANT,
                             pad_val=0, p=p)
    except TypeError:
        return A.Perspective(scale=(0.02, 0.06), p=p)


def _handwriting_extra(cfg: AugConfig) -> list[A.BasicTransform]:
    return [
        _perspective(cfg.p_perspective),
        A.ElasticTransform(alpha=30, sigma=6, p=cfg.p_elastic),
        A.OneOf([_shadow()], p=cfg.p_shadow),
        PaperTexture(strength=0.18, p=cfg.p_paper_texture),
        InkFade(p=cfg.p_ink_fade),
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
    elif cfg.tier == "standard":
        ops = [*_base_geometric(cfg), *_photometric(cfg), *_normalize(cfg)]
    elif cfg.tier == "handwriting":
        ops = [
            *_base_geometric(cfg),
            *_handwriting_extra(cfg),
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
