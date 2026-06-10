"""Per-language handwriting font pools with coverage filtering and cached ImageFonts."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache

from PIL import ImageFont

from .assets import load_or_build_coverage
from .config import FontConfig

logger = logging.getLogger(__name__)

_HELP = ("No usable {lang} handwriting fonts in {dirs}.\n"
         "Put .ttf/.otf there, or run:  python scripts/fetch_fonts.py")


@dataclass(frozen=True)
class FontEntry:
    path: str
    coverage: float
    covered: frozenset = field(default_factory=frozenset)

    def filter(self, text: str) -> str:
        return "".join(c for c in text if c in self.covered)


@lru_cache(maxsize=512)
def _open_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


class FontBank:
    def __init__(self, cfg: FontConfig, ru_charset: str, en_charset: str):
        self.cfg = cfg
        # ``extra`` = the OTHER language's charset: recorded into ``covered`` so code-switched
        # glyphs survive filtering, but NOT into ``coverage`` -> pool membership is unchanged.
        self.pools = {"ru": self._build(cfg.ru_font_dirs, ru_charset, "ru", en_charset),
                      "en": self._build(cfg.en_font_dirs, en_charset, "en", ru_charset)}

    def _build(self, dirs, charset, lang, extra_charset=""):
        cov = load_or_build_coverage(dirs, charset, extra_charset)
        entries = [FontEntry(p, float(i["coverage"]), frozenset(i["covered"]))
                   for p, i in cov.items() if i["coverage"] >= self.cfg.min_glyph_coverage]
        if not entries:
            raise RuntimeError(_HELP.format(lang=lang, dirs=list(dirs)) +
                               f"  ({len(cov)} found, all below {self.cfg.min_glyph_coverage})")
        weights = [e.coverage for e in entries] if self.cfg.weight_by_coverage else None
        logger.info("FontBank[%s]: %d fonts (%.2f-%.2f)", lang, len(entries),
                    min(e.coverage for e in entries), max(e.coverage for e in entries))
        return entries, weights

    def __len__(self) -> int:
        return sum(len(e) for e, _ in self.pools.values())

    def n(self, lang: str) -> int:
        return len(self.pools[lang][0])

    def sample(self, rng, lang: str, require=None) -> FontEntry:
        import numpy as np
        entries, weights = self.pools[lang]
        if require:                              # keep only fonts covering the required glyphs
            req = frozenset(require)
            keep = [i for i, e in enumerate(entries) if req <= e.covered]
            if keep:                             # fall back to the full pool if none qualify
                entries = [entries[i] for i in keep]
                weights = [weights[i] for i in keep] if weights is not None else None
        if weights is None:
            return entries[int(rng.integers(0, len(entries)))]
        w = np.asarray(weights, dtype=np.float64)
        return entries[int(rng.choice(len(entries), p=w / w.sum()))]

    def get(self, entry: FontEntry, size_px: int) -> ImageFont.FreeTypeFont:
        return _open_font(entry.path, int(size_px))

    def warm_cache(self) -> None:
        lo, hi = self.cfg.sizes_px
        for entries, _ in self.pools.values():
            for e in entries:
                for s in {int(lo), int(hi), int((lo + hi) // 2)}:
                    try:
                        _open_font(e.path, s)
                    except Exception:
                        pass
