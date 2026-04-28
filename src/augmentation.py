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

import random
from dataclasses import dataclass, field
from pathlib import Path
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
    box_expand_height_fraction: float = 0.0
    mean: tuple[float, float, float] = field(default_factory=lambda: IMAGENET_MEAN)
    std: tuple[float, float, float] = field(default_factory=lambda: IMAGENET_STD)


class SyntheticPaperAugmenter:
    """Paper-style pre-augmentation copied from the U-Net binarization notebook.

    It is intentionally separate from the regular DBNet augmenter so the old
    real-image training path keeps the same behavior.
    """

    def __init__(self, image_size: int, table_backgrounds_dir: str | Path | None = None):
        self.image_size = image_size
        self.table_backgrounds = self._load_table_backgrounds(table_backgrounds_dir)
        self.geometric = self._build_geometric()
        self.shadow_transforms = self._build_shadow_transforms()

    @staticmethod
    def _load_table_backgrounds(table_backgrounds_dir: str | Path | None) -> list[Path]:
        if not table_backgrounds_dir:
            return []
        root = Path(table_backgrounds_dir)
        if not root.exists():
            return []
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        return sorted(path for path in root.iterdir() if path.suffix.lower() in exts)

    @staticmethod
    def _affine_no_crop() -> A.BasicTransform:
        try:
            return A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(-0.0625, 0.0625),
                rotate=(-30, 30),
                fit_output=True,
                border_mode=cv2.BORDER_CONSTANT,
                fill=255,
                fill_mask=0,
                p=1,
            )
        except TypeError:
            return A.ShiftScaleRotate(
                shift_limit=(-0.0625, 0.0625),
                scale_limit=(-0.1, 0.1),
                rotate_limit=(-30, 30),
                border_mode=cv2.BORDER_CONSTANT,
                value=255,
                mask_value=0,
                p=1,
            )

    def _build_geometric(self) -> A.Compose:
        return A.Compose(
            [
                A.OneOf(
                    [
                        A.Compose(
                            [
                                A.RandomRotate90(p=1),
                                A.HorizontalFlip(p=0.5),
                            ],
                            p=0.4,
                        ),
                        A.Compose(
                            [
                                self._affine_no_crop(),
                                A.HorizontalFlip(p=0.5),
                            ],
                            p=0.6,
                        ),
                    ],
                    p=1,
                ),
                A.LongestMaxSize(
                    max_size=self.image_size,
                    interpolation=cv2.INTER_LINEAR,
                    mask_interpolation=cv2.INTER_NEAREST,
                    p=1,
                ),
                self._pad_white_with_empty_mask(),
            ],
            keypoint_params=_keypoint_params(),
            p=1,
        )

    def _pad_white_with_empty_mask(self) -> A.BasicTransform:
        try:
            return A.PadIfNeeded(
                min_height=self.image_size,
                min_width=self.image_size,
                border_mode=cv2.BORDER_CONSTANT,
                fill=255,
                fill_mask=0,
                p=1,
            )
        except TypeError:
            return A.PadIfNeeded(
                min_height=self.image_size,
                min_width=self.image_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=255,
                mask_value=0,
                p=1,
            )


    @staticmethod
    def _build_shadow_transforms() -> A.Compose:
        all_image = A.Compose(
            [
                A.OneOf(
                    [
                        A.RandomShadow(
                            shadow_roi=(0, 0, 1, 1),
                            num_shadows_limit=(1, 1),
                            shadow_dimension=6,
                            shadow_intensity_range=(0.08, 0.16),
                            p=1,
                        ),
                        A.RandomShadow(
                            shadow_roi=(0, 0, 1, 1),
                            num_shadows_limit=(1, 1),
                            shadow_dimension=3,
                            shadow_intensity_range=(0.10, 0.22),
                            p=1,
                        ),
                        A.RandomSunFlare(
                            flare_roi=(0, 0, 1, 1),
                            src_radius=120,
                            src_color=(200, 200, 200),
                            p=1,
                        ),
                        A.RandomBrightnessContrast(
                            brightness_limit=(-0.2, 0.2),
                            contrast_limit=(-0.2, 0.2),
                            brightness_by_max=False,
                            p=1,
                        ),
                        A.Defocus(
                            radius=(1, 2),
                            alias_blur=(0.01, 0.03),
                            p=1,
                        ),
                    ],
                    p=1,
                ),
            ],
            p=0.65,
        )
        squares = [(0, 0, 0.5, 0.5), (0.5, 0, 1, 0.5), (0, 0.5, 0.5, 1), (0.5, 0.5, 1, 1)]
        transforms_squares_list: list[A.BasicTransform] = []
        for roi in squares:
            transforms_squares_list.append(
                A.RandomShadow(
                    shadow_roi=roi,
                    num_shadows_limit=(1, 1),
                    shadow_dimension=7,
                    shadow_intensity_range=(0.06, 0.16),
                    p=1,
                )
            )
            transforms_squares_list.append(
                A.RandomSunFlare(
                    flare_roi=roi,
                    src_radius=100,
                    src_color=(200, 200, 200),
                    p=1,
                )
            )
            transforms_squares_list.append(A.Defocus(radius=(1, 2), alias_blur=(0.01, 0.03), p=1))
        transforms_squares = A.OneOf(
            [A.SomeOf(transforms_squares_list, n=1, p=1)] * 5
            + [A.SomeOf(transforms_squares_list, n=2, p=1)] * 3
            + [A.SomeOf(transforms_squares_list, n=3, p=1)] * 2,
            p=1,
        )
        return A.OneOf(
            [
                A.Compose([all_image], p=0.4),
                A.Compose([transforms_squares], p=0.5),
                A.Compose([transforms_squares, all_image], p=0.1),
            ],
            p=1,
        )

    def _random_table_background(self, height: int, width: int) -> np.ndarray | None:
        if not self.table_backgrounds:
            return None
        path = random.choice(self.table_backgrounds)
        bg = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bg is None:
            return None
        bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
        bg_h, bg_w = bg.shape[:2]
        scale = max(height / max(bg_h, 1), width / max(bg_w, 1))
        bg = cv2.resize(bg, (max(width, int(bg_w * scale)), max(height, int(bg_h * scale))))
        y = random.randint(0, max(0, bg.shape[0] - height))
        x = random.randint(0, max(0, bg.shape[1] - width))
        return bg[y:y + height, x:x + width].copy()

    def _replace_background(
        self,
        image: np.ndarray,
        text_mask: np.ndarray,
        page_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        table = self._random_table_background(*image.shape[:2])
        if table is None:
            return image
        alpha = random.uniform(0.6, 0.9)
        table = (table.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)

        if page_mask is not None:
            keep_mask = page_mask > 0
            result = table.copy()
            result[keep_mask] = image[keep_mask]
            return result

        coords = np.column_stack(np.where(text_mask > 0))
        if len(coords) == 0:
            return image
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        pad = 50
        h, w = image.shape[:2]
        y_min = max(0, int(y_min) - pad)
        y_max = min(h, int(y_max) + pad)
        x_min = max(0, int(x_min) - pad)
        x_max = min(w, int(x_max) + pad)

        page_mask = np.zeros((h, w), dtype=np.uint8)
        page_mask[y_min:y_max, x_min:x_max] = 1
        page_mask = cv2.dilate(page_mask, np.ones((7, 7), np.uint8)).astype(bool)

        result = table.copy()
        result[page_mask] = image[page_mask]
        return result

    @staticmethod
    def _add_background_artifacts(image: np.ndarray, text_mask: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        result = image.copy()
        coords = np.column_stack(np.where(text_mask > 0))
        if len(coords) == 0:
            return result

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        pad = 50
        x_min = max(0, int(x_min) - pad)
        x_max = min(w, int(x_max) + pad)
        y_min = max(0, int(y_min) - pad)
        y_max = min(h, int(y_max) + pad)

        allowed = np.ones((h, w), dtype=np.uint8)
        allowed[y_min:y_max, x_min:x_max] = 0

        for _ in range(random.randint(1, 3)):
            shape_type = random.choice(["circle", "rectangle"])
            color = random.randint(0, 100)
            if shape_type == "circle":
                r = random.randint(150, 500)
                if h <= 2 * r + 1 or w <= 2 * r + 1:
                    continue
                for _ in range(100):
                    x = random.randint(r, w - r - 1)
                    y = random.randint(r, h - r - 1)
                    y1, y2 = y - r, y + r + 1
                    x1, x2 = x - r, x + r + 1
                    patch = allowed[y1:y2, x1:x2]
                    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
                    circle_mask = (xx * xx + yy * yy) <= r * r
                    if patch.shape == circle_mask.shape and np.all(patch[circle_mask] == 1):
                        cv2.circle(result, (x, y), r, (color, color, color), -1)
                        break
            else:
                rect_w = random.randint(150, 500)
                rect_h = random.randint(150, 500)
                if h <= rect_h + 1 or w <= rect_w + 1:
                    continue
                for _ in range(100):
                    x1 = random.randint(0, w - rect_w - 1)
                    y1 = random.randint(0, h - rect_h - 1)
                    x2 = x1 + rect_w
                    y2 = y1 + rect_h
                    if np.all(allowed[y1:y2, x1:x2] == 1):
                        result[y1:y2, x1:x2] = (color, color, color)
                        break
        return result

    def __call__(
        self,
        image: np.ndarray,
        polygons: Sequence[np.ndarray],
    ) -> tuple[np.ndarray, list[np.ndarray], list[bool]]:
        polygons = [np.asarray(p, dtype=np.float32).reshape(-1, 2) for p in polygons]
        sizes = [len(p) for p in polygons]
        flat_kp = np.concatenate(polygons, axis=0) if sizes else np.zeros((0, 2), np.float32)
        page_mask = np.full(image.shape[:2], 255, dtype=np.uint8)

        out = self.geometric(image=image, mask=page_mask, keypoints=flat_kp.tolist())
        image_t = out["image"]
        page_mask_t = out["mask"]
        text_mask_t = self._text_mask_from_image(image_t)
        kp_t = np.asarray(out["keypoints"], dtype=np.float32).reshape(-1, 2) if out["keypoints"] else np.zeros((0, 2), np.float32)

        image_t = self.shadow_transforms(image=image_t)["image"]
        image_t = self._replace_background(image_t, text_mask_t, page_mask=page_mask_t)
        if random.random() < 0.6:
            image_t = self._add_background_artifacts(image_t, text_mask_t)

        h, w = image_t.shape[:2]
        polys_t: list[np.ndarray] = []
        keep: list[bool] = []
        cursor = 0
        for n in sizes:
            pts = kp_t[cursor:cursor + n]
            cursor += n
            if len(pts) < 3:
                keep.append(False)
                continue
            pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
            if min(pts[:, 0].max() - pts[:, 0].min(), pts[:, 1].max() - pts[:, 1].min()) < 2:
                keep.append(False)
                continue
            polys_t.append(pts)
            keep.append(True)
        return image_t, polys_t, keep

    @staticmethod
    def _text_mask_from_image(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        return (gray < 245).astype(np.uint8) * 255


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
