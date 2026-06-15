"""Orchestrator: language -> text -> font -> ink -> paper -> composite -> degrade -> resize."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image

import dataclasses

from .backgrounds import PaperBackground
from .config import CorpusConfig, FontConfig, SynthConfig, _coerce
from .corpus import TextSampler
from .effects import Compositor, EffectsPipeline
from .fonts import FontBank
from .render import LineRenderer
from .rng import chance, randint, scale_p, uniform

_WS = re.compile(r"\s+")


def _apply_overrides(cfg: SynthConfig, node) -> SynthConfig:
    """Patch a SynthConfig from a mapping/OmegaConf node, sub-block by sub-block. Only the
    fields present in the node are changed; everything else keeps its (stage-1) default."""
    for name in ("corpus", "font", "render", "paper", "effects", "neighbors", "output"):
        sub = node.get(name) if hasattr(node, "get") else None
        if not sub:
            continue
        cur = getattr(cfg, name)
        fields = {f: _coerce(sub[f]) for f in cur.__dataclass_fields__ if f in sub}
        cfg = dataclasses.replace(cfg, **{name: dataclasses.replace(cur, **fields)})
    return cfg


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
                  cache_dir=None, synth_overrides=None, **cfg_kwargs):
        """Build from per-language folders. ``*_text_weights`` (len == dirs) bias which
        folder is sampled more often. Only explicitly passed knobs override CorpusConfig
        defaults. Empty text dirs -> built-in word fallback. ``synth_overrides`` is an
        optional mapping (e.g. a yaml ``synth`` node) whose sub-blocks — corpus / render /
        effects / neighbors / paper / output — patch the config (used for stage-2)."""
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
        if synth_overrides is not None:
            cfg = _apply_overrides(cfg, synth_overrides)
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
        ncfg = self.cfg.neighbors
        if ncfg.p_neighbor > 0 and chance(rng, scale_p(ncfg.p_neighbor, t)):
            paper, oy = self._add_neighbors(paper, max(0, oy), ink.height, rng, t)
        rgb = self.compositor.blend(paper, ink, (max(0, ox), max(0, oy)), rng, t)
        img = Image.fromarray(self.effects(np.asarray(rgb), rng, t))
        return (img, text) if img.height >= self.cfg.output.min_height_px else None

    def _neighbor_ink(self, rng, t):
        """Render a distractor line (independent text + font); returns an RGBA ink crop."""
        cfg = self.cfg.neighbors
        text, lang = self.sampler.sample(rng, t)
        entry = self.fonts.sample(rng, lang)
        text = _WS.sub(" ", entry.filter(_WS.sub(" ", text))).strip()[: cfg.max_chars].strip()
        if len(text) < 1:
            return None
        font = self.fonts.get(entry, randint(rng, self.cfg.font.sizes_px))
        nink, meta = self.renderer.render(text, font, rng, t)
        if meta.get("empty") or nink.width < 2 or nink.height < 2:
            return None
        return nink

    def _add_neighbors(self, paper, oy, ink_h, rng, t):
        """Grow the canvas just enough to host a thin sliver of a neighbour line hugging the
        main text above and/or below (a small gap, as when a line detector crops slightly into
        the next line). The added rows are filled by replicating the paper's edge so the
        background stays continuous. Returns (paper, new_oy). The neighbour is a distractor:
        the label is unchanged."""
        cfg = self.cfg.neighbors
        W, H = paper.size

        def _strip(side):
            nink = self._neighbor_ink(rng, t)
            if nink is None:
                return None
            vpx = max(3, min(nink.height, int(uniform(rng, cfg.visible_frac) * ink_h)))
            box = (0, nink.height - vpx, nink.width, nink.height) if side == "top" else (0, 0, nink.width, vpx)
            crop = nink.crop(box)
            if crop.width > W:                                  # random horizontal window
                x0 = int(rng.integers(0, crop.width - W + 1))
                crop = crop.crop((x0, 0, x0 + W, crop.height))
            return crop, randint(rng, (1, max(1, ink_h // 12)))   # (sliver, gap to the main text)

        sides = (["top", "bottom"] if chance(rng, cfg.p_both_sides)
                 else (["top"] if rng.random() < 0.5 else ["bottom"]))
        strips = {s: v for s in sides if (v := _strip(s)) is not None}
        if not strips:
            return paper, oy

        add_t = strips["top"][0].height + strips["top"][1] if "top" in strips else 0
        add_b = strips["bottom"][0].height + strips["bottom"][1] if "bottom" in strips else 0
        arr = np.asarray(paper.convert("RGB"))
        parts = ([np.repeat(arr[:1], add_t, axis=0)] if add_t else []) + [arr] + \
                ([np.repeat(arr[-1:], add_b, axis=0)] if add_b else [])
        base = Image.fromarray(np.concatenate(parts, axis=0)).convert("RGBA")

        if "top" in strips:                                     # sliver sits gap-px above the text
            crop, _ = strips["top"]
            x = int(rng.integers(0, max(1, W - crop.width + 1)))
            base.alpha_composite(crop, (x, 0))
        if "bottom" in strips:                                  # ... and/or gap-px below it
            crop, gap = strips["bottom"]
            x = int(rng.integers(0, max(1, W - crop.width + 1)))
            base.alpha_composite(crop, (x, base.height - crop.height))
        return base.convert("RGB"), oy + add_t

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
