"""Render a string into a tight-cropped RGBA ink layer with handwriting variation."""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from .config import RenderConfig
from .rng import chance, eased_centered, eased_uniform, lerp, uniform


class LineRenderer:
    def __init__(self, cfg: RenderConfig):
        self.cfg = cfg

    def render(self, text: str, font, rng, t: float = 1.0):
        cfg = self.cfg
        ink = tuple(cfg.ink_colors[int(rng.integers(0, len(cfg.ink_colors)))])
        pencil = chance(rng, cfg.p_pencil)
        base_alpha = uniform(rng, cfg.pencil_alpha if pencil else cfg.ink_alpha)
        grain_amt = cfg.stroke_grain * (1.4 if pencil else 1.0)

        ascent, descent = font.getmetrics()
        glyph_h = max(1, ascent + descent)
        pad = cfg.pad_px + 4
        wob_max = cfg.baseline_wobble_px[1]

        space_min = cfg.space_min_frac * glyph_h
        advances, jit = [], []
        for ch in text:
            a = float(font.getlength(ch))
            if ch == " ":
                a = max(a, space_min)          # guarantee a word gap (fonts vary)
            advances.append(a)
            jit.append(1.0 + eased_centered(rng, cfg.spacing_jitter, t))
        total_w = int(sum(a * j for a, j in zip(advances, jit))) + 2 * pad + glyph_h
        canvas_h = int(glyph_h * 1.6 + 2 * pad + wob_max)
        baseline_y = int(pad + ascent + 0.3 * glyph_h)

        canvas = Image.new("RGBA", (max(1, total_w), max(1, canvas_h)), (0, 0, 0, 0))
        wob = self._wobble(total_w, eased_uniform(rng, cfg.baseline_wobble_px, t), rng)

        x = float(pad)
        for ch, adv, j in zip(text, advances, jit):
            if ch != " " and adv > 0:
                tile, base = self._glyph_tile(ch, font, ink, ascent, descent)
                scale = lerp(1.0, uniform(rng, cfg.size_jitter), t)
                if abs(scale - 1.0) > 1e-3:
                    tile = tile.resize((max(1, int(tile.width * scale)),
                                        max(1, int(tile.height * scale))), Image.BICUBIC)
                    base *= scale
                rot = lerp(0.0, uniform(rng, cfg.per_glyph_rot_deg), t)
                if abs(rot) > 0.2:
                    h0 = tile.height
                    tile = tile.rotate(rot, expand=True, resample=Image.BICUBIC)
                    base += (tile.height - h0) / 2.0
                wy = int(round(baseline_y + wob[min(int(x), total_w - 1)] - base))
                canvas.alpha_composite(tile, (int(round(x)), max(0, wy)))
            x += adv * j

        canvas = self._shear(canvas, eased_centered(rng, cfg.slant_deg, t), baseline_y)
        canvas = self._rotate(canvas, eased_centered(rng, cfg.line_rotate_deg, t))
        canvas = self._apply_alpha(canvas, base_alpha, grain_amt, rng)

        bbox = canvas.getbbox()
        if bbox is None:
            return Image.new("RGBA", (1, 1), (0, 0, 0, 0)), {"empty": True}
        p = cfg.pad_px
        x0, y0, x1, y1 = bbox
        crop = canvas.crop((max(0, x0 - p), max(0, y0 - p),
                            min(canvas.width, x1 + p), min(canvas.height, y1 + p)))
        return crop, {"ink": ink, "pencil": pencil, "empty": False}

    @staticmethod
    def _glyph_tile(ch, font, ink, ascent, descent):
        adv = max(1, int(math.ceil(font.getlength(ch))))
        pad = 3
        tile = Image.new("RGBA", (adv + 2 * pad, ascent + descent + 2 * pad), (0, 0, 0, 0))
        ImageDraw.Draw(tile).text((pad, pad + ascent), ch, font=font,
                                  fill=(ink[0], ink[1], ink[2], 255), anchor="ls")
        return tile, float(pad + ascent)

    @staticmethod
    def _wobble(width, amp, rng):
        if amp <= 0 or width < 2:
            return np.zeros(max(1, width), dtype=np.float32)
        k = max(2, width // 60)
        ctrl = rng.uniform(-amp, amp, size=k)
        return np.interp(np.arange(width), np.linspace(0, width - 1, k), ctrl).astype(np.float32)

    @staticmethod
    def _shear(canvas, slant_deg, baseline_y):
        if abs(slant_deg) < 0.2:
            return canvas
        shear = math.tan(math.radians(slant_deg))
        w, h = canvas.size
        c = shear * baseline_y if shear > 0 else 0.0
        return canvas.transform((w + int(math.ceil(abs(shear) * h)), h), Image.AFFINE,
                                (1.0, shear, -c, 0.0, 1.0, 0.0), resample=Image.BICUBIC)

    @staticmethod
    def _rotate(canvas, deg):
        # rotate the transparent ink layer -> empty corners stay transparent (paper fills them)
        if abs(deg) < 0.2:
            return canvas
        return canvas.rotate(deg, expand=True, resample=Image.BICUBIC)

    @staticmethod
    def _apply_alpha(canvas, base_alpha, grain_amt, rng):
        arr = np.asarray(canvas).astype(np.float32)
        a = arr[:, :, 3] * base_alpha
        if grain_amt > 0:
            a *= rng.uniform(1.0 - grain_amt, 1.0, size=a.shape).astype(np.float32)
        arr[:, :, 3] = np.clip(a, 0, 255)
        return Image.fromarray(arr.astype(np.uint8), "RGBA")
