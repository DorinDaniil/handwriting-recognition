"""Procedural paper substrates: plain / ruled (линейка) / grid (клетка) / real-scan crop.

Grid + ruled notebook paper is the dominant background in Russian school
handwriting, so it gets first-class treatment. Lines are drawn at a sampled
spacing with a *random phase* so the text doesn't always sit exactly on a rule.
Everything is cheap PIL/numpy; an optional pre-rendered pool (see assets) can
replace live drawing if the grid cost ever dominates throughput.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .assets import RealPaperPool
from .config import PaperConfig
from .rng import chance, eased_uniform, randint, scale_p, uniform


class PaperBackground:
    def __init__(self, cfg: PaperConfig):
        self.cfg = cfg
        self.pool = RealPaperPool(cfg.real_paper_dir)

    def make(self, size_wh: tuple[int, int], rng, t: float = 1.0) -> Image.Image:
        w, h = int(size_wh[0]), int(size_wh[1])
        cfg = self.cfg
        kinds = ["plain", "ruled", "grid"]
        weights = [cfg.p_plain, scale_p(cfg.p_ruled, t), scale_p(cfg.p_grid, t)]
        if len(self.pool):
            kinds.append("real"); weights.append(scale_p(cfg.p_real_crop, t))
        kind = kinds[int(rng.choice(len(kinds), p=np.asarray(weights) / np.sum(weights)))]

        if kind == "real":
            img = Image.fromarray(self.pool.sample_crop((w, h), rng))
        else:
            img = Image.fromarray(self._paper_field(w, h, rng))
            if kind == "ruled":
                self._draw_ruled(img, rng)
            elif kind == "grid":
                self._draw_grid(img, rng)

        if chance(rng, scale_p(cfg.p_margin_line, t)):
            self._draw_margin(img, rng)
        v = eased_uniform(rng, cfg.vignette, t)
        if v > 0.01:
            img = self._vignette(img, v)
        return img

    # ----------------------------------------------------------- internals

    def _paper_field(self, w: int, h: int, rng) -> np.ndarray:
        color = self.cfg.paper_colors[int(rng.integers(0, len(self.cfg.paper_colors)))]
        base = np.empty((h, w, 3), dtype=np.float32)
        base[:] = color
        if self.cfg.fiber_noise > 0:
            noise = rng.normal(0.0, self.cfg.fiber_noise * 255.0, size=(h, w, 1)).astype(np.float32)
            base += noise
        return np.clip(base, 0, 255).astype(np.uint8)

    def _line_overlay(self, img: Image.Image) -> tuple[Image.Image, ImageDraw.ImageDraw]:
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        return overlay, ImageDraw.Draw(overlay)

    def _blend(self, img: Image.Image, overlay: Image.Image) -> None:
        img.paste(Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB"), (0, 0))

    def _draw_ruled(self, img, rng) -> None:
        w, h = img.size
        sp = randint(rng, self.cfg.rule_spacing_px)
        col = self.cfg.rule_colors[int(rng.integers(0, len(self.cfg.rule_colors)))]
        a = int(uniform(rng, self.cfg.rule_alpha) * 255)
        phase = int(rng.integers(0, sp))
        ov, d = self._line_overlay(img)
        for y in range(phase, h, sp):
            d.line([(0, y), (w, y)], fill=(col[0], col[1], col[2], a), width=1)
        self._blend(img, ov)

    def _draw_grid(self, img, rng) -> None:
        w, h = img.size
        sp = randint(rng, self.cfg.grid_spacing_px)
        col = self.cfg.rule_colors[int(rng.integers(0, len(self.cfg.rule_colors)))]
        a = int(uniform(rng, self.cfg.rule_alpha) * 255)
        px, py = int(rng.integers(0, sp)), int(rng.integers(0, sp))
        ov, d = self._line_overlay(img)
        for y in range(py, h, sp):
            d.line([(0, y), (w, y)], fill=(col[0], col[1], col[2], a), width=1)
        for x in range(px, w, sp):
            d.line([(x, 0), (x, h)], fill=(col[0], col[1], col[2], a), width=1)
        self._blend(img, ov)

    def _draw_margin(self, img, rng) -> None:
        w, h = img.size
        col = self.cfg.margin_color
        a = int(uniform(rng, self.cfg.rule_alpha) * 255)
        x = int(uniform(rng, (0.06, 0.16)) * w)
        ov, d = self._line_overlay(img)
        d.line([(x, 0), (x, h)], fill=(col[0], col[1], col[2], a), width=1)
        self._blend(img, ov)

    @staticmethod
    def _vignette(img: Image.Image, strength: float) -> Image.Image:
        w, h = img.size
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        r2 = ((xx - cx) / (cx + 1e-3)) ** 2 + ((yy - cy) / (cy + 1e-3)) ** 2
        factor = (1.0 - strength * np.clip(r2, 0, 1))[:, :, None]
        out = np.asarray(img, dtype=np.float32) * factor
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")
