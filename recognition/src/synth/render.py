"""Render a string into a tight-cropped RGBA *ink layer* with handwriting variation.

The realism that matters for HTR comes from breaking the "every glyph identical"
look of plain font rendering:
  * per-glyph size jitter and a small rotation,
  * a smooth (low-frequency) baseline wobble across the line,
  * inter-glyph spacing jitter (incl. slight negative = pseudo-connection),
  * a line-level slant (shear),
  * ink colour + a dry-pen alpha grain (pencil vs pen).

Output is RGBA on a transparent canvas so the compositor can drop it onto any
paper. All heavy pixel math is vectorised numpy; per-glyph work is a handful of
small PIL ops, keeping throughput high enough for on-the-fly DataLoader use.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from .config import RenderConfig
from .rng import chance, eased_uniform, lerp, uniform


class LineRenderer:
    def __init__(self, cfg: RenderConfig):
        self.cfg = cfg

    def render(self, text: str, font, rng, t: float = 1.0):
        """Return ``(rgba: PIL.Image, meta: dict)``. ``rgba`` is tight-cropped to the
        ink. Raises nothing on empty ink — returns a 1x1 transparent image instead."""
        cfg = self.cfg
        ink = tuple(cfg.ink_colors[int(rng.integers(0, len(cfg.ink_colors)))])
        pencil = chance(rng, cfg.p_pencil)
        base_alpha = uniform(rng, cfg.pencil_alpha if pencil else cfg.ink_alpha)
        grain_amt = cfg.stroke_grain * (1.4 if pencil else 1.0)

        ascent, descent = font.getmetrics()
        glyph_h = max(1, ascent + descent)
        pad = cfg.pad_px + 4
        wob_max = cfg.baseline_wobble_px[1]

        # advances + spacing jitter (curriculum-eased toward neutral at t=0)
        advances, jit = [], []
        for ch in text:
            advances.append(float(font.getlength(ch)) if ch != " " else float(font.getlength(" ")))
            jit.append(1.0 + eased_uniform(rng, cfg.spacing_jitter, t))
        total_w = int(sum(a * j for a, j in zip(advances, jit))) + 2 * pad + glyph_h
        canvas_h = int(glyph_h * 1.6 + 2 * pad + wob_max)
        baseline_y = int(pad + ascent + 0.3 * glyph_h)

        canvas = Image.new("RGBA", (max(1, total_w), max(1, canvas_h)), (0, 0, 0, 0))
        wob = self._wobble(total_w, eased_uniform(rng, cfg.baseline_wobble_px, t), rng)

        x = float(pad)
        for ch, adv, j in zip(text, advances, jit):
            if ch != " " and adv > 0:
                tile, base_in_tile = self._glyph_tile(ch, font, ink, ascent, descent)
                scale = lerp(1.0, uniform(rng, cfg.size_jitter), t)
                if abs(scale - 1.0) > 1e-3:
                    tile = tile.resize((max(1, int(tile.width * scale)),
                                        max(1, int(tile.height * scale))), Image.BICUBIC)
                    base_in_tile *= scale
                rot = lerp(0.0, uniform(rng, cfg.per_glyph_rot_deg), t)
                if abs(rot) > 0.2:
                    h0 = tile.height
                    tile = tile.rotate(rot, expand=True, resample=Image.BICUBIC)
                    base_in_tile += (tile.height - h0) / 2.0   # expand centres the content
                wy = int(round(baseline_y + wob[min(int(x), total_w - 1)] - base_in_tile))
                canvas.alpha_composite(tile, (int(round(x)), max(0, wy)))
            x += adv * j

        canvas = self._shear(canvas, eased_uniform(rng, cfg.slant_deg, t), baseline_y)
        canvas = self._apply_alpha(canvas, base_alpha, grain_amt, rng)

        bbox = canvas.getbbox()
        if bbox is None:
            return Image.new("RGBA", (1, 1), (0, 0, 0, 0)), {"empty": True}
        p = cfg.pad_px
        x0, y0, x1, y1 = bbox
        crop = canvas.crop((max(0, x0 - p), max(0, y0 - p),
                            min(canvas.width, x1 + p), min(canvas.height, y1 + p)))
        return crop, {"ink": ink, "pencil": pencil, "empty": False}

    # ----------------------------------------------------------- internals

    @staticmethod
    def _glyph_tile(ch, font, ink, ascent, descent):
        """One glyph on its own transparent tile; returns (tile, baseline_row)."""
        adv = max(1, int(math.ceil(font.getlength(ch))))
        pad = 3
        h = ascent + descent + 2 * pad
        tile = Image.new("RGBA", (adv + 2 * pad, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(tile)
        baseline = pad + ascent
        d.text((pad, baseline), ch, font=font, fill=(ink[0], ink[1], ink[2], 255), anchor="ls")
        return tile, float(baseline)

    @staticmethod
    def _wobble(width: int, amp: float, rng) -> np.ndarray:
        """Smooth low-frequency vertical baseline offset across the line."""
        if amp <= 0 or width < 2:
            return np.zeros(max(1, width), dtype=np.float32)
        k = max(2, width // 60)
        ctrl = rng.uniform(-amp, amp, size=k)
        xs = np.linspace(0, width - 1, k)
        return np.interp(np.arange(width), xs, ctrl).astype(np.float32)

    @staticmethod
    def _shear(canvas: Image.Image, slant_deg: float, baseline_y: int) -> Image.Image:
        if abs(slant_deg) < 0.2:
            return canvas
        shear = math.tan(math.radians(slant_deg))
        w, h = canvas.size
        extra = int(math.ceil(abs(shear) * h))
        new_w = w + extra
        # source_x = x - shear*(baseline_y - y); positive slant leans the top rightwards
        c = shear * baseline_y if shear > 0 else 0.0
        out = canvas.transform((new_w, h), Image.AFFINE, (1.0, shear, -c, 0.0, 1.0, 0.0),
                               resample=Image.BICUBIC)
        return out

    @staticmethod
    def _apply_alpha(canvas: Image.Image, base_alpha: float, grain_amt: float, rng) -> Image.Image:
        arr = np.asarray(canvas).astype(np.float32)
        a = arr[:, :, 3] * base_alpha
        if grain_amt > 0:
            grain = rng.uniform(1.0 - grain_amt, 1.0, size=a.shape).astype(np.float32)
            a = a * grain
        arr[:, :, 3] = np.clip(a, 0, 255)
        return Image.fromarray(arr.astype(np.uint8), "RGBA")
