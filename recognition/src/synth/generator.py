"""Orchestrator: language -> text -> font -> ink -> paper -> composite -> degrade -> resize."""
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
    """Letterbox into size×size (e.g. for a square TrOCR input)."""
    w, h = img.size
    if w / max(1, h) > max_aspect:
        h = max(1, int(round(w / max_aspect)))
    scale = size / max(w, h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    canvas = Image.new("RGB", (size, size), tuple(pad_color))
    canvas.paste(img.resize((nw, nh), Image.BILINEAR), ((size - nw) // 2, (size - nh) // 2))
    return canvas


def resize_to_min_side(img: Image.Image, min_side: int = 224, max_side: int | None = None) -> Image.Image:
    """Scale so the shorter side == min_side (aspect kept, no padding)."""
    w, h = img.size
    scale = min_side / max(1, min(w, h))
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    if max_side and max(nw, nh) > max_side:
        s = max_side / max(nw, nh)
        nw, nh = max(1, round(nw * s)), max(1, round(nh * s))
    return img.resize((nw, nh), Image.BILINEAR)


class HandwrittenLineGenerator:
    def __init__(self, cfg: SynthConfig | None = None, *, sampler=None, fonts=None,
                 renderer=None, paper=None, compositor=None, effects=None):
        self.cfg = cfg or SynthConfig()
        self.sampler = sampler or TextSampler(self.cfg.corpus, self.cfg.ru_charset, self.cfg.en_charset)
        self.fonts = fonts or FontBank(self.cfg.font, self.cfg.ru_charset, self.cfg.en_charset)
        self.renderer = renderer or LineRenderer(self.cfg.render)
        self.paper = paper or PaperBackground(self.cfg.paper)
        self.compositor = compositor or Compositor(self.cfg.effects)
        self.effects = effects or EffectsPipeline(self.cfg.effects)

    @classmethod
    def from_dirs(cls, ru_text_dirs=(), en_text_dirs=(),
                  ru_font_dirs=("assets/fonts_ru",), en_font_dirs=("assets/fonts_en",), *,
                  ru_text_weights=(), en_text_weights=(),
                  p_ru=None, len_chars=None, p_hyphenate=None, glob="*.txt",
                  cache_dir=None, **cfg_kwargs):
        """Build from per-language folders. ``*_text_weights`` (len == dirs) bias which
        folder is sampled more often. Only explicitly passed knobs override CorpusConfig
        defaults. Empty text dirs -> built-in word fallback."""
        def _t(x):
            return (str(x),) if isinstance(x, (str, Path)) else tuple(str(p) for p in x)
        corpus = dict(ru_text_dirs=_t(ru_text_dirs), en_text_dirs=_t(en_text_dirs),
                      ru_text_weights=tuple(ru_text_weights), en_text_weights=tuple(en_text_weights),
                      glob=glob, cache_dir=cache_dir)
        for key, val in (("p_ru", p_ru), ("p_hyphenate", p_hyphenate)):
            if val is not None:
                corpus[key] = float(val)
        if len_chars is not None:
            corpus["len_chars"] = tuple(len_chars)
        cfg = SynthConfig(corpus=CorpusConfig(**corpus),
                          font=FontConfig(ru_font_dirs=_t(ru_font_dirs), en_font_dirs=_t(en_font_dirs)),
                          **cfg_kwargs)
        return cls(cfg)

    def difficulty(self, step: int) -> float:
        if not self.cfg.curriculum:
            return 1.0
        return float(min(max(step, 0) / max(1, self.cfg.warmup_steps), 1.0))

    def warm_cache(self) -> None:
        self.fonts.warm_cache()

    def render_line(self, rng, step: int = 0):
        t = self.difficulty(step)
        for _ in range(4):
            out = self._try_once(rng, t)
            if out is not None:
                return out
        return self._fallback(rng)

    def sample(self, rng, step: int = 0):
        img, text = self.render_line(rng, step)
        return resize_to_min_side(img, self.cfg.output.min_side, self.cfg.output.max_side), text

    __call__ = sample

    def _try_once(self, rng, t):
        text, lang = self.sampler.sample(rng, t)
        entry = self.fonts.sample(rng, lang)
        text = _WS.sub(" ", entry.filter(_WS.sub(" ", text))).strip()
        if len(text) < 1:
            return None
        font = self.fonts.get(entry, randint(rng, self.cfg.font.sizes_px))
        ink, meta = self.renderer.render(text, font, rng, t)
        if meta.get("empty") or ink.height < 2 or ink.width < 2:
            return None
        mfrac = self.cfg.output.margin_frac
        mw = max(2, int(ink.height * uniform(rng, mfrac)))
        mh = max(2, int(ink.height * uniform(rng, mfrac)))
        paper = self.paper.make((ink.width + 2 * mw, ink.height + 2 * mh), rng, t)
        ox = mw + int(uniform(rng, (-0.4, 0.4)) * mw)
        oy = mh + int(uniform(rng, (-0.4, 0.4)) * mh)
        rgb = self.compositor.blend(paper, ink, (max(0, ox), max(0, oy)), rng, t)
        img = Image.fromarray(self.effects(np.asarray(rgb), rng, t))
        return (img, text) if img.height >= self.cfg.output.min_height_px else None

    def _fallback(self, rng):
        lang = "ru" if self.fonts.n("ru") else "en"
        entry = max(self.fonts.pools[lang][0], key=lambda e: e.coverage)
        text = entry.filter("пример текста" if lang == "ru" else "example text") or "abc"
        font = self.fonts.get(entry, int(np.mean(self.cfg.font.sizes_px)))
        ink, _ = self.renderer.render(text, font, rng, 0.0)
        paper = Image.new("RGB", (ink.width + 28, ink.height + 20),
                          self.cfg.paper.paper_colors[0]).convert("RGBA")
        paper.alpha_composite(ink, (14, 10))
        return resize_to_min_side(paper.convert("RGB"), self.cfg.output.min_side,
                                  self.cfg.output.max_side), text
