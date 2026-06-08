"""Compositing (ink → paper) + capture-degradation pipeline.

``EffectsPipeline`` prefers **albumentations** (already a project dependency, and
installed on the training box) for the rich geometric/photometric transforms,
following the same version-guard pattern as ``detection/src/augmentation.py``
(kwargs were renamed across albumentations releases). When albumentations is
absent (e.g. a quick local run) it transparently falls back to a numpy/PIL
subset so the generator still produces degraded samples for inspection.

Two custom ops have no albumentations equivalent and are always applied via
numpy: a sinusoidal **baseline curve** (row remap) and a smooth **illumination**
gradient. Geometry is line-safe: small rotation, elastic, grid, perspective —
never flips or 90° rotations. Output is uint8 RGB; **no normalization happens
here** — the TrOCR processor is the only normalizer.
"""
from __future__ import annotations

import io
import math

import numpy as np
from PIL import Image, ImageFilter

from .config import EffectsConfig
from .rng import chance, scale_p, uniform

try:
    import albumentations as A
    _HAS_ALBU = True
except Exception:  # pragma: no cover
    A = None
    _HAS_ALBU = False


# ============================ compositor ============================

class Compositor:
    """Place the ink layer on paper with optional bleed and back-page show-through."""

    def __init__(self, cfg: EffectsConfig):
        self.cfg = cfg

    def blend(self, paper: Image.Image, ink_rgba: Image.Image,
              offset: tuple[int, int], rng, t: float = 1.0) -> Image.Image:
        cfg = self.cfg
        base = paper.convert("RGBA")
        ink = ink_rgba
        bleed = uniform(rng, cfg.ink_bleed_px)
        if bleed > 0.05:
            ink = ink.filter(ImageFilter.GaussianBlur(bleed))
        if chance(rng, scale_p(cfg.p_show_through, t)):
            ghost = ink.transpose(Image.FLIP_LEFT_RIGHT).filter(ImageFilter.GaussianBlur(1.2))
            g = np.asarray(ghost).astype(np.float32)
            g[:, :, 3] *= 0.08
            gx = max(0, offset[0] - int(0.5 * ink.width))
            base.alpha_composite(Image.fromarray(g.astype(np.uint8), "RGBA"), (gx, offset[1]))
        base.alpha_composite(ink, (int(offset[0]), int(offset[1])))
        return base.convert("RGB")


# ============================ effects ============================

class EffectsPipeline:
    """Geometry → photometric degradation. ``p_*`` are gated with the *explicit*
    RNG (decorrelated across workers) and scaled by the curriculum ``t``; the
    transform *parameters* come from albumentations' own RNG (seeded per worker)."""

    def __init__(self, cfg: EffectsConfig):
        self.cfg = cfg
        self.backend = "albumentations" if _HAS_ALBU else "fallback"
        if _HAS_ALBU:
            self._geo = self._build_geo()
            self._photo = self._build_photo()

    def __call__(self, img: np.ndarray, rng, t: float = 1.0) -> np.ndarray:
        img = self._geometry(img, rng, t)
        img = self._photometric(img, rng, t)
        return img

    # ----------------------------------------------------- geometry

    def _geometry(self, img: np.ndarray, rng, t: float) -> np.ndarray:
        cfg = self.cfg
        if chance(rng, scale_p(cfg.p_baseline_curve, t)):
            img = self._baseline_curve(img, rng)
        if _HAS_ALBU:
            for p, key in ((cfg.p_elastic, "elastic"), (cfg.p_grid_distort, "grid"),
                           (cfg.p_perspective, "persp"), (cfg.p_affine_rotate, "rot")):
                if chance(rng, scale_p(p, t)):
                    img = self._geo[key](image=img)["image"]
        else:
            if chance(rng, scale_p(cfg.p_affine_rotate, t)):
                img = self._rotate_pil(img, rng)
        return img

    def _build_geo(self) -> dict:
        cfg = self.cfg
        return {
            "elastic": A.ElasticTransform(alpha=float(np.mean(cfg.elastic_alpha)),
                                          sigma=float(np.mean(cfg.elastic_sigma)), p=1.0),
            "grid": A.GridDistortion(num_steps=5, distort_limit=0.2, p=1.0),
            "persp": A.Perspective(scale=cfg.perspective_scale, fit_output=True, p=1.0),
            "rot": A.Affine(rotate=cfg.affine_rotate_deg, fit_output=True,
                            mode=0, cval=255, p=1.0),
        }

    # ----------------------------------------------------- photometric

    def _photometric(self, img: np.ndarray, rng, t: float) -> np.ndarray:
        cfg = self.cfg
        if chance(rng, scale_p(cfg.p_illumination, t)):
            img = self._illumination(img, rng)
        if _HAS_ALBU:
            order = (("blur", cfg.p_blur), ("motion", cfg.p_motion_blur),
                     ("gnoise", cfg.p_gauss_noise), ("iso", cfg.p_iso_noise),
                     ("bc", cfg.p_brightness_contrast), ("gamma", cfg.p_gamma),
                     ("jpeg", cfg.p_jpeg), ("down", cfg.p_downscale))
            for key, p in order:
                if chance(rng, scale_p(p, t)):
                    img = self._photo[key](image=img)["image"]
        else:
            img = self._photometric_fallback(img, rng, t)
        return np.ascontiguousarray(img)

    def _build_photo(self) -> dict:
        cfg = self.cfg
        # version-guarded kwargs (renamed across albumentations releases)
        try:
            gnoise = A.GaussNoise(std_range=(0.02, 0.12), p=1.0)
        except TypeError:
            gnoise = A.GaussNoise(var_limit=(10.0, 60.0), p=1.0)
        try:
            jpeg = A.ImageCompression(quality_range=cfg.jpeg_quality, p=1.0)
        except TypeError:
            jpeg = A.ImageCompression(quality_lower=cfg.jpeg_quality[0],
                                      quality_upper=cfg.jpeg_quality[1], p=1.0)
        try:
            down = A.Downscale(scale_range=cfg.downscale_range, p=1.0)
        except TypeError:
            down = A.Downscale(scale_min=cfg.downscale_range[0],
                               scale_max=cfg.downscale_range[1], p=1.0)
        return {
            "blur": A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            "motion": A.MotionBlur(blur_limit=5, p=1.0),
            "gnoise": gnoise,
            "iso": A.ISONoise(p=1.0),
            "bc": A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            "gamma": A.RandomGamma(gamma_limit=(80, 120), p=1.0),
            "jpeg": jpeg,
            "down": down,
        }

    # ----------------------------------------------------- custom numpy ops

    @staticmethod
    def _baseline_curve(img: np.ndarray, rng) -> np.ndarray:
        h, w = img.shape[:2]
        amp = float(rng.uniform(1.0, 3.0))
        freq = float(rng.uniform(0.5, 2.0))
        phase = float(rng.uniform(0, 2 * math.pi))
        dy = (amp * np.sin(2 * math.pi * freq * np.arange(w) / max(1, w) + phase)).astype(int)
        rows = np.clip(np.arange(h)[:, None] + dy[None, :], 0, h - 1)
        cols = np.broadcast_to(np.arange(w), (h, w))
        return img[rows, cols]

    @staticmethod
    def _illumination(img: np.ndarray, rng) -> np.ndarray:
        h, w = img.shape[:2]
        s = float(rng.uniform(0.08, 0.22))
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        ang = float(rng.uniform(0, 2 * math.pi))
        g = (np.cos(ang) * (xx / w) + np.sin(ang) * (yy / h))
        g = (g - g.min()) / (np.ptp(g) + 1e-6)
        factor = (1.0 - s + 2 * s * g)[:, :, None]
        return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    # ----------------------------------------------------- fallback ops

    @staticmethod
    def _rotate_pil(img: np.ndarray, rng) -> np.ndarray:
        ang = float(rng.uniform(-3.0, 3.0))
        pil = Image.fromarray(img).rotate(ang, expand=True, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
        return np.asarray(pil)

    def _photometric_fallback(self, img: np.ndarray, rng, t: float) -> np.ndarray:
        cfg = self.cfg
        pil = Image.fromarray(img)
        if chance(rng, scale_p(cfg.p_blur, t)):
            pil = pil.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.5, 1.4))))
        arr = np.asarray(pil).astype(np.float32)
        if chance(rng, scale_p(cfg.p_brightness_contrast, t)):
            b = float(rng.uniform(-0.15, 0.15)) * 255
            c = float(rng.uniform(0.85, 1.15))
            arr = (arr - 128) * c + 128 + b
        if chance(rng, scale_p(cfg.p_gamma, t)):
            g = float(rng.uniform(0.8, 1.25))
            arr = np.clip(arr, 0, 255)
            arr = ((arr / 255.0) ** g) * 255.0
        if chance(rng, scale_p(cfg.p_gauss_noise, t)):
            arr = arr + rng.normal(0, float(rng.uniform(4, 18)), size=arr.shape)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        if chance(rng, scale_p(cfg.p_downscale, t)):
            f = float(rng.uniform(*cfg.downscale_range))
            h, w = arr.shape[:2]
            small = Image.fromarray(arr).resize((max(1, int(w * f)), max(1, int(h * f))), Image.BILINEAR)
            arr = np.asarray(small.resize((w, h), Image.BILINEAR))
        if chance(rng, scale_p(cfg.p_jpeg, t)):
            q = int(rng.integers(cfg.jpeg_quality[0], cfg.jpeg_quality[1] + 1))
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, format="JPEG", quality=q)
            buf.seek(0)
            arr = np.asarray(Image.open(buf).convert("RGB"))
        return arr
