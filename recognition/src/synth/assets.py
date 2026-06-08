"""Asset discovery & caching: font files, glyph-coverage manifest, real-paper pool.

Glyph coverage is the single most important guard for Cyrillic synthesis — most
free "handwriting" fonts are Latin-only or cover only uppercase Cyrillic, and
ё/й/ъ/щ are routinely missing. We read the font ``cmap`` with fontTools (exact,
pure-python) and fall back to a PIL rasterization heuristic only if fontTools is
unavailable. The result is cached to ``<font_dir>/_coverage.json`` keyed by file
mtime+size and the active charset, so the (slow) probe runs once.
"""
from __future__ import annotations

import hashlib
import json
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageFont

logger = logging.getLogger(__name__)

FONT_EXTS = {".ttf", ".otf", ".ttc"}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
_MANIFEST_NAME = "_coverage.json"


# ------------------------- font discovery -------------------------

def scan_font_files(font_dirs) -> list[Path]:
    """Recursively collect font files under the given directories (sorted, deduped)."""
    out: list[Path] = []
    seen: set[Path] = set()
    for d in font_dirs:
        root = Path(d)
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix.lower() in FONT_EXTS and p.resolve() not in seen:
                seen.add(p.resolve())
                out.append(p)
    return out


# ------------------------- glyph coverage -------------------------

def font_charset(path: Path) -> set[str]:
    """Set of unicode characters a font can render. Tries fontTools, then PIL."""
    try:
        from fontTools.ttLib import TTFont, TTCollection

        if path.suffix.lower() == ".ttc":
            coll = TTCollection(str(path), lazy=True)
            chars: set[str] = set()
            for f in coll.fonts:
                cmap = f.getBestCmap() or {}
                chars |= {chr(cp) for cp in cmap}
            coll.close()
            return chars
        f = TTFont(str(path), fontNumber=0, lazy=True)
        cmap = f.getBestCmap() or {}
        chars = {chr(cp) for cp in cmap}
        f.close()
        return chars
    except Exception as e:  # pragma: no cover - fallback path
        logger.debug("fontTools failed on %s (%s); using PIL heuristic", path.name, e)
        return _font_charset_pil(path)


def _font_charset_pil(path: Path, probe_charset: str | None = None) -> set[str]:
    """Heuristic fallback: a glyph is "covered" if it rasterizes to non-empty ink.
    Imperfect (a font's .notdef box can read as covered) — only used without fontTools."""
    from .config import DEFAULT_CHARSET
    probe = probe_charset or DEFAULT_CHARSET
    try:
        font = ImageFont.truetype(str(path), 40)
    except Exception:
        return set()
    covered: set[str] = set()
    for ch in probe:
        if ch.isspace():
            covered.add(ch)
            continue
        try:
            if font.getmask(ch).getbbox() is not None:
                covered.add(ch)
        except Exception:
            pass
    return covered


def _charset_hash(charset: str) -> str:
    return hashlib.md5(charset.encode("utf-8")).hexdigest()[:12]


def _file_key(path: Path) -> str:
    st = path.stat()
    return f"{int(st.st_mtime)}:{st.st_size}"


def load_or_build_coverage(font_dirs, charset: str, refresh: bool = False) -> dict[str, dict]:
    """Return ``{font_path: {"coverage": float, "covered": "<chars∩charset>"}}``.

    Cached to ``<first existing font_dir>/_coverage.json``; entries are reused when
    the file mtime/size and the active charset are unchanged.
    """
    files = scan_font_files(font_dirs)
    chash = _charset_hash(charset)
    active = set(charset)

    manifest_dir = next((Path(d) for d in font_dirs if Path(d).exists()), Path(font_dirs[0]))
    manifest_path = manifest_dir / _MANIFEST_NAME

    cache: dict = {}
    if manifest_path.exists() and not refresh:
        try:
            cache = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            cache = {}
    if cache.get("charset_hash") != chash:
        cache = {"charset_hash": chash, "fonts": {}}
    cache.setdefault("fonts", {})

    result: dict[str, dict] = {}
    dirty = False
    for path in files:
        key = str(path)
        fkey = _file_key(path)
        entry = cache["fonts"].get(key)
        if entry is None or entry.get("fkey") != fkey:
            covered = font_charset(path) & active
            cov = len(covered) / max(1, len(active))
            entry = {"fkey": fkey, "coverage": cov, "covered": "".join(sorted(covered))}
            cache["fonts"][key] = entry
            dirty = True
        result[key] = {"coverage": entry["coverage"], "covered": entry["covered"]}

    if dirty:
        try:
            manifest_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        except Exception as e:  # pragma: no cover
            logger.warning("could not write coverage manifest %s: %s", manifest_path, e)
    return result


# ------------------------- real paper pool -------------------------

@lru_cache(maxsize=64)
def _load_rgb(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


class RealPaperPool:
    """Random crops from a directory of real scanned-paper photos (optional)."""

    def __init__(self, directory: str | None):
        self.paths: list[str] = []
        if directory and Path(directory).exists():
            self.paths = [str(p) for p in sorted(Path(directory).rglob("*"))
                          if p.suffix.lower() in IMG_EXTS]

    def __len__(self) -> int:
        return len(self.paths)

    def sample_crop(self, size_wh: tuple[int, int], rng) -> np.ndarray:
        """Return an (H, W, 3) uint8 crop resized to ``size_wh`` = (W, H)."""
        w, h = size_wh
        src = _load_rgb(self.paths[int(rng.integers(0, len(self.paths)))])
        sh, sw = src.shape[:2]
        # take a window of the target aspect, then resize
        ar = w / max(1, h)
        cw = min(sw, max(8, int(sh * ar)))
        ch = min(sh, max(8, int(cw / ar)))
        x0 = int(rng.integers(0, max(1, sw - cw + 1)))
        y0 = int(rng.integers(0, max(1, sh - ch + 1)))
        crop = src[y0:y0 + ch, x0:x0 + cw]
        return np.asarray(Image.fromarray(crop).resize((w, h), Image.BILINEAR))
