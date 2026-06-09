"""Compositing (ink -> paper) + capture degradation (albumentations, numpy/PIL fallback)."""
from __future__ import annotations

import io
import math

import numpy as np
from PIL import Image, ImageFilter

from .config import EffectsConfig
from .rng import chance, scale_p, uniform

try:
    import albumentations as A
    import cv2
    _HAS_ALBU = True
    _BORDER = cv2.BORDER_REPLICATE          # extend paper at edges, never white/black
except Exception:
    A = None
    cv2 = None
    _HAS_ALBU = False
    _BORDER = 1


class Compositor:
    def __init__(self, cfg: EffectsConfig):
        self.cfg = cfg

    def blend(self, paper, ink_rgba, offset, rng, t: float = 1.0) -> Image.Image:
        cfg = self.cfg
        base = paper.convert("RGBA")
        ink = ink_rgba
        bleed = uniform(rng, cfg.ink_bleed_px)
        if bleed > 0.05:
            ink = ink.filter(ImageFilter.GaussianBlur(bleed))
        if chance(rng, scale_p(cfg.p_show_through, t)):
            ghost = ink.transpose(Image.FLIP_LEFT_RIGHT).filter(ImageFilter.GaussianBlur(1.2))
            g = np.asarray(ghost).astype(np.float32); g[:, :, 3] *= 0.08
            base.alpha_composite(Image.fromarray(g.astype(np.uint8), "RGBA"),
                                 (max(0, offset[0] - int(0.5 * ink.width)), offset[1]))
        base.alpha_composite(ink, (int(offset[0]), int(offset[1])))
        return base.convert("RGB")


class EffectsPipeline:
    def __init__(self, cfg: EffectsConfig):
        self.cfg = cfg
        self.backend = "albumentations" if _HAS_ALBU else "fallback"
        if _HAS_ALBU:
            self._geo = self._build_geo()
            self._photo = self._build_photo()

    def __call__(self, img: np.ndarray, rng, t: float = 1.0) -> np.ndarray:
        return self._photometric(self._geometry(img, rng, t), rng, t)

    # geometry (line slant is in the renderer; here only paper-preserving warps)

    def _geometry(self, img, rng, t):
        cfg = self.cfg
        if chance(rng, scale_p(cfg.p_baseline_curve, t)):
            img = self._baseline_curve(img, rng)
        if _HAS_ALBU:
            for p, key in ((cfg.p_elastic, "elastic"), (cfg.p_grid_distort, "grid"),
                           (cfg.p_perspective, "persp")):
                tr = self._geo.get(key)
                if tr is not None and chance(rng, scale_p(p, t)):
                    img = tr(image=img)["image"]
        return img

    def _build_geo(self):
        cfg = self.cfg
        geo = {}
        try:
            geo["elastic"] = A.ElasticTransform(alpha=float(np.mean(cfg.elastic_alpha)),
                                                sigma=float(np.mean(cfg.elastic_sigma)),
                                                border_mode=_BORDER, p=1.0)
        except TypeError:
            geo["elastic"] = A.ElasticTransform(alpha=float(np.mean(cfg.elastic_alpha)),
                                                sigma=float(np.mean(cfg.elastic_sigma)), p=1.0)
        geo["grid"] = None      # normalized=True keeps cells in-frame (no text pushed out)
        for extra in ({"normalized": True, "border_mode": _BORDER}, {"border_mode": _BORDER}, {}):
            try:
                geo["grid"] = A.GridDistortion(num_steps=5, distort_limit=0.15, p=1.0, **extra)
                break
            except TypeError:
                continue
        geo["persp"] = None     # fit_output=True keeps the whole line (no crop); replicate border = paper
        for kw in ("border_mode", "pad_mode"):
            try:
                geo["persp"] = A.Perspective(scale=cfg.perspective_scale, fit_output=True,
                                             p=1.0, **{kw: _BORDER})
                break
            except TypeError:
                continue
        return geo

    # photometric

    def _photometric(self, img, rng, t):
        cfg = self.cfg
        if chance(rng, scale_p(cfg.p_illumination, t)):
            img = self._illumination(img, rng)
        if _HAS_ALBU:
            order = (("blur", cfg.p_blur), ("motion", cfg.p_motion_blur), ("gnoise", cfg.p_gauss_noise),
                     ("iso", cfg.p_iso_noise), ("bc", cfg.p_brightness_contrast), ("gamma", cfg.p_gamma),
                     ("jpeg", cfg.p_jpeg), ("down", cfg.p_downscale))
            for key, p in order:
                if chance(rng, scale_p(p, t)):
                    img = self._photo[key](image=img)["image"]
            return np.ascontiguousarray(img)
        return self._photometric_fallback(img, rng, t)

    def _build_photo(self):
        cfg = self.cfg
        try:    # kwargs renamed across albumentations versions
            gnoise = A.GaussNoise(std_range=(0.02, 0.07), p=1.0)
        except TypeError:
            gnoise = A.GaussNoise(var_limit=(8.0, 35.0), p=1.0)
        try:
            jpeg = A.ImageCompression(quality_range=cfg.jpeg_quality, p=1.0)
        except TypeError:
            jpeg = A.ImageCompression(quality_lower=cfg.jpeg_quality[0],
                                      quality_upper=cfg.jpeg_quality[1], p=1.0)
        try:
            down = A.Downscale(scale_range=cfg.downscale_range, p=1.0)
        except TypeError:
            down = A.Downscale(scale_min=cfg.downscale_range[0], scale_max=cfg.downscale_range[1], p=1.0)
        return {
            "blur": A.GaussianBlur(blur_limit=(3, 3), p=1.0),
            "motion": A.MotionBlur(blur_limit=3, p=1.0),
            "gnoise": gnoise, "iso": A.ISONoise(p=1.0),
            "bc": A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1.0),
            "gamma": A.RandomGamma(gamma_limit=(85, 115), p=1.0), "jpeg": jpeg, "down": down,
        }

    # custom numpy ops (no albumentations equivalent, used in both backends)

    @staticmethod
    def _baseline_curve(img, rng):
        h, w = img.shape[:2]
        dy = (rng.uniform(1.0, 3.0) * np.sin(2 * math.pi * rng.uniform(0.5, 2.0)
              * np.arange(w) / max(1, w) + rng.uniform(0, 2 * math.pi))).astype(int)
        rows = np.clip(np.arange(h)[:, None] + dy[None, :], 0, h - 1)
        return img[rows, np.broadcast_to(np.arange(w), (h, w))]

    @staticmethod
    def _illumination(img, rng):
        h, w = img.shape[:2]
        s = float(rng.uniform(0.08, 0.22))
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        ang = float(rng.uniform(0, 2 * math.pi))
        g = np.cos(ang) * (xx / w) + np.sin(ang) * (yy / h)
        g = (g - g.min()) / (np.ptp(g) + 1e-6)
        return np.clip(img.astype(np.float32) * (1.0 - s + 2 * s * g)[:, :, None], 0, 255).astype(np.uint8)

    def _photometric_fallback(self, img, rng, t):
        cfg = self.cfg
        pil = Image.fromarray(img)
        if chance(rng, scale_p(cfg.p_blur, t)):
            pil = pil.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.4, 1.0))))
        arr = np.asarray(pil).astype(np.float32)
        if chance(rng, scale_p(cfg.p_brightness_contrast, t)):
            arr = (arr - 128) * float(rng.uniform(0.85, 1.15)) + 128 + float(rng.uniform(-0.15, 0.15)) * 255
        if chance(rng, scale_p(cfg.p_gamma, t)):
            arr = ((np.clip(arr, 0, 255) / 255.0) ** float(rng.uniform(0.8, 1.25))) * 255.0
        if chance(rng, scale_p(cfg.p_gauss_noise, t)):
            arr = arr + rng.normal(0, float(rng.uniform(3, 11)), size=arr.shape)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        if chance(rng, scale_p(cfg.p_downscale, t)):
            f = float(rng.uniform(*cfg.downscale_range)); h, w = arr.shape[:2]
            small = Image.fromarray(arr).resize((max(1, int(w * f)), max(1, int(h * f))), Image.BILINEAR)
            arr = np.asarray(small.resize((w, h), Image.BILINEAR))
        if chance(rng, scale_p(cfg.p_jpeg, t)):
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, format="JPEG",
                                      quality=int(rng.integers(cfg.jpeg_quality[0], cfg.jpeg_quality[1] + 1)))
            buf.seek(0); arr = np.asarray(Image.open(buf).convert("RGB"))
        return arr
