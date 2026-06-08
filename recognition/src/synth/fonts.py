"""Font pool management: coverage-filtered sampling + cached ``ImageFont`` objects.

``FontBank`` holds only paths + per-font covered-char sets, which are picklable,
so a built bank can be shipped to DataLoader workers; each worker then lazily
reopens (and LRU-caches) the actual ``ImageFont`` objects — opening a TTF on
every ``render`` call is the #1 avoidable cost in on-the-fly generation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache

from PIL import ImageFont

from .assets import load_or_build_coverage
from .config import FontConfig

logger = logging.getLogger(__name__)

_FONT_HELP = (
    "No usable Cyrillic handwriting fonts found in {dirs}.\n"
    "Drop .ttf/.otf files there, or run:  python scripts/fetch_fonts.py\n"
    "Good free sources: fonts with Cyrillic 'handwriting' style — "
    "Caveat, Marck Script, Bad Script, Neucha, Pangolin, Pacifico (Google Fonts), "
    "plus localfonts.eu / fontesk.com / fontspace.com (handwriting+cyrillic)."
)


@dataclass(frozen=True)
class FontEntry:
    path: str
    coverage: float
    covered: frozenset = field(default_factory=frozenset)

    def can_render(self, text: str) -> bool:
        return all(c in self.covered for c in text)

    def filter(self, text: str) -> str:
        """Keep only renderable characters (label stays in sync with the pixels)."""
        return "".join(c for c in text if c in self.covered)


@lru_cache(maxsize=256)
def _open_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


class FontBank:
    """A coverage-filtered pool of handwriting fonts."""

    def __init__(self, cfg: FontConfig, charset: str):
        self.cfg = cfg
        self.charset = charset
        cov = load_or_build_coverage(cfg.font_dirs, charset)
        entries: list[FontEntry] = []
        for path, info in cov.items():
            if info["coverage"] >= cfg.min_glyph_coverage:
                entries.append(FontEntry(path=path,
                                         coverage=float(info["coverage"]),
                                         covered=frozenset(info["covered"])))
        self.entries = entries
        if not entries:
            n_found = len(cov)
            extra = (f" ({n_found} font(s) found but all below "
                     f"min_glyph_coverage={cfg.min_glyph_coverage:.2f})" if n_found else "")
            raise RuntimeError(_FONT_HELP.format(dirs=list(cfg.font_dirs)) + extra)
        # sampling weights: optionally favour higher-coverage fonts
        self._weights = ([e.coverage for e in entries]
                         if cfg.weight_by_coverage else None)
        logger.info("FontBank: %d fonts (coverage %.2f–%.2f)", len(entries),
                    min(e.coverage for e in entries), max(e.coverage for e in entries))

    def __len__(self) -> int:
        return len(self.entries)

    def sample(self, rng) -> FontEntry:
        import numpy as np
        if self._weights is None:
            return self.entries[int(rng.integers(0, len(self.entries)))]
        w = np.asarray(self._weights, dtype=np.float64)
        w /= w.sum()
        return self.entries[int(rng.choice(len(self.entries), p=w))]

    def get(self, entry: FontEntry, size_px: int) -> ImageFont.FreeTypeFont:
        return _open_font(entry.path, int(size_px))

    def warm_cache(self) -> None:
        """Pre-open every (font, size_bound) so the first samples don't pay for it.
        Call once per worker in ``worker_init_fn``."""
        lo, hi = self.cfg.sizes_px
        for e in self.entries:
            for s in {int(lo), int(hi), int((lo + hi) // 2)}:
                try:
                    _open_font(e.path, s)
                except Exception:
                    pass
