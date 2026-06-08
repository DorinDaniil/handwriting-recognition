"""The orchestrator: assemble corpus → font → ink → paper → composite → degrade.

``HandwrittenLineGenerator.sample(rng, step)`` returns ``(PIL.Image, str)`` ready
for the TrOCR processor (letterboxed to a square if ``output.keep_aspect``).
``render_line`` returns the *natural* (variable-aspect) line — handy for visual
inspection. ``difficulty(step)`` drives the curriculum ``t∈[0,1]`` that every
stage reads. Components are pluggable (pass your own to the constructor) so a
future GAN/diffusion ink source (Tier 2) can drop in behind the same contract.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image

from .backgrounds import PaperBackground
from .config import CorpusConfig, FontConfig, SynthConfig
from .corpus import TextSampler
from .effects import Compositor, EffectsPipeline
from .fonts import FontBank
from .render import LineRenderer
from .rng import randint, uniform

_WS = re.compile(r"\s+")


def fit_to_square(img: Image.Image, size: int, pad_color=(255, 255, 255),
                  max_aspect: float = 8.0) -> Image.Image:
    """Letterbox ``img`` into a ``size×size`` canvas, preserving aspect (no squash).

    Use the SAME function on real line crops at inference time so train/test
    preprocessing matches (cf. ``detection/src/utils.preprocess_image_pil``)."""
    w, h = img.size
    if w / max(1, h) > max_aspect:                       # clamp pathological wide lines
        h = max(1, int(round(w / max_aspect)))
    scale = size / max(w, h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), tuple(pad_color))
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return canvas


class HandwrittenLineGenerator:
    def __init__(self, cfg: SynthConfig | None = None, *, sampler=None, fonts=None,
                 renderer=None, paper=None, compositor=None, effects=None):
        self.cfg = cfg or SynthConfig()
        self.sampler = sampler or TextSampler(self.cfg.corpus, self.cfg.charset)
        self.fonts = fonts or FontBank(self.cfg.font, self.cfg.charset)
        self.renderer = renderer or LineRenderer(self.cfg.render)
        self.paper = paper or PaperBackground(self.cfg.paper)
        self.compositor = compositor or Compositor(self.cfg.effects)
        self.effects = effects or EffectsPipeline(self.cfg.effects)

    @classmethod
    def from_dirs(cls, text_dirs, font_dirs=("assets/fonts",), *,
                  len_chars=(8, 50), p_hyphenate=0.15, glob="*.txt", **cfg_kwargs):
        """One-liner: build straight from folder(s) of .txt and folder(s) of fonts.

            gen = HandwrittenLineGenerator.from_dirs(
                text_dirs=["/data/books", "/data/notes"],   # str or list of folders
                font_dirs="assets/fonts",
                len_chars=(15, 45),    # line length in characters (varies the image size)
                p_hyphenate=0.3,       # fraction of lines ending mid-word with '-' (перенос)
            )

        ``len_chars`` / ``p_hyphenate`` / ``glob`` configure the corpus; any other
        keyword goes to ``SynthConfig`` (e.g. ``curriculum=False``, ``warmup_steps=8000``).
        For full control build ``SynthConfig`` yourself and pass it to the constructor."""
        def _as_tuple(x):
            return (str(x),) if isinstance(x, (str, Path)) else tuple(str(p) for p in x)
        cfg = SynthConfig(
            corpus=CorpusConfig(text_dirs=_as_tuple(text_dirs), len_chars=tuple(len_chars),
                                p_hyphenate=float(p_hyphenate), glob=glob),
            font=FontConfig(font_dirs=_as_tuple(font_dirs)),
            **cfg_kwargs)
        return cls(cfg)

    # ----------------------------------------------------------- curriculum

    def difficulty(self, step: int) -> float:
        if not self.cfg.curriculum:
            return 1.0
        return float(min(max(step, 0) / max(1, self.cfg.warmup_steps), 1.0))

    def warm_cache(self) -> None:
        """Pre-open fonts (call once per DataLoader worker)."""
        self.fonts.warm_cache()

    # ----------------------------------------------------------- sampling

    def render_line(self, rng, step: int = 0):
        """Return ``(PIL.Image RGB, str)`` — the natural, variable-aspect line."""
        t = self.difficulty(step)
        for _ in range(4):
            out = self._try_once(rng, t)
            if out is not None:
                return out
        return self._fallback(rng)

    def sample(self, rng, step: int = 0):
        """Return ``(PIL.Image RGB, str)`` ready for the processor."""
        img, text = self.render_line(rng, step)
        out = self.cfg.output
        if out.keep_aspect:
            img = fit_to_square(img, out.proc_size, out.pad_color, out.max_aspect)
        return img, text

    __call__ = sample

    # ----------------------------------------------------------- internals

    def _try_once(self, rng, t: float):
        text = _WS.sub(" ", self.sampler.sample(rng, t)).strip()
        entry = self.fonts.sample(rng)
        text = _WS.sub(" ", entry.filter(text)).strip()
        if len(text) < 1:
            return None
        size = randint(rng, self.cfg.font.sizes_px)
        font = self.fonts.get(entry, size)
        ink, meta = self.renderer.render(text, font, rng, t)
        if meta.get("empty") or ink.height < 2 or ink.width < 2:
            return None

        mx = int(size * uniform(rng, (0.3, 1.4)))
        my = int(size * uniform(rng, (0.3, 0.9)))
        W, H = ink.width + 2 * mx, ink.height + 2 * my
        paper = self.paper.make((W, H), rng, t)
        ox = mx + int(uniform(rng, (-0.3, 0.3)) * mx)
        oy = my + int(uniform(rng, (-0.3, 0.3)) * my)
        rgb = self.compositor.blend(paper, ink, (max(0, ox), max(0, oy)), rng, t)
        arr = self.effects(np.asarray(rgb), rng, t)
        img = Image.fromarray(arr)
        if img.height < self.cfg.output.min_height_px:
            return None
        return img, text

    def _fallback(self, rng):
        """Never-fail path: best-coverage font, plain paper, no degradation."""
        entry = max(self.fonts.entries, key=lambda e: e.coverage)
        text = entry.filter("пример текста") or "текст"
        font = self.fonts.get(entry, int(np.mean(self.cfg.font.sizes_px)))
        ink, _ = self.renderer.render(text, font, rng, 0.0)
        mx, my = 16, 10
        paper = Image.new("RGB", (ink.width + 2 * mx, ink.height + 2 * my),
                          self.cfg.paper.paper_colors[0])
        rgb = paper.convert("RGBA")
        rgb.alpha_composite(ink, (mx, my))
        img = rgb.convert("RGB")
        if self.cfg.output.keep_aspect:
            img = fit_to_square(img, self.cfg.output.proc_size,
                                self.cfg.output.pad_color, self.cfg.output.max_aspect)
        return img, text
