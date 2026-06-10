"""Font discovery, glyph-coverage manifest (cached), and a real-paper crop pool."""
from __future__ import annotations

import hashlib
import json
import logging
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageFont

logger = logging.getLogger(__name__)

FONT_EXTS = {".ttf", ".otf", ".ttc"}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
_MANIFEST_NAME = "_coverage.json"


def scan_font_files(font_dirs) -> list[Path]:
    out, seen = [], set()
    for d in font_dirs:
        root = Path(d)
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix.lower() in FONT_EXTS and p.resolve() not in seen:
                seen.add(p.resolve()); out.append(p)
    return out


def font_charset(path: Path) -> set[str]:
    """Characters a font can render (fontTools cmap, PIL fallback)."""
    try:
        from fontTools.ttLib import TTCollection, TTFont
        if path.suffix.lower() == ".ttc":
            coll = TTCollection(str(path), lazy=True)
            chars = set()
            for f in coll.fonts:
                chars |= {chr(cp) for cp in (f.getBestCmap() or {})}
            coll.close()
            return chars
        f = TTFont(str(path), fontNumber=0, lazy=True)
        chars = {chr(cp) for cp in (f.getBestCmap() or {})}
        f.close()
        return chars
    except Exception as e:
        logger.debug("fontTools failed on %s (%s); PIL fallback", path.name, e)
        return _font_charset_pil(path)


def _font_charset_pil(path: Path, probe: str | None = None) -> set[str]:
    from .config import RU_CHARSET, EN_CHARSET
    probe = probe or (RU_CHARSET + EN_CHARSET)
    try:
        font = ImageFont.truetype(str(path), 40)
    except Exception:
        return set()
    covered = set()
    for ch in probe:
        if ch.isspace():
            covered.add(ch)
        else:
            try:
                if font.getmask(ch).getbbox() is not None:
                    covered.add(ch)
            except Exception:
                pass
    return covered


def _file_key(path: Path) -> str:
    st = path.stat()
    return f"{int(st.st_mtime)}:{st.st_size}"


def load_or_build_coverage(font_dirs, charset: str, extra_charset: str = "",
                           refresh: bool = False) -> dict[str, dict]:
    """Return {path: {coverage, covered}} for fonts, cached per (file, charset).

    ``coverage`` is measured over ``charset`` (drives the threshold and pool). ``covered``
    is the glyphs the font renders across ``charset`` + ``extra_charset`` — passing the other
    language as ``extra_charset`` lets code-switched insertions (e.g. Latin in a RU font)
    survive glyph filtering, without changing ``coverage`` or pool membership."""
    files = scan_font_files(font_dirs)
    chash = hashlib.md5((charset + "||" + extra_charset).encode("utf-8")).hexdigest()[:12]
    active, probe = set(charset), set(charset) | set(extra_charset)
    cache = Path(tempfile.gettempdir()) / "synth_fonts"   # writable; fonts dir may be read-only
    cache.mkdir(parents=True, exist_ok=True)
    dkey = hashlib.md5("|".join(sorted(map(str, font_dirs))).encode()).hexdigest()[:16]
    manifest_path = cache / f"{dkey}.json"

    cache = {}
    if manifest_path.exists() and not refresh:
        try:
            cache = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            cache = {}
    if cache.get("charset_hash") != chash:
        cache = {"charset_hash": chash, "fonts": {}}
    cache.setdefault("fonts", {})

    result, dirty = {}, False
    for path in files:
        key, fkey = str(path), _file_key(path)
        entry = cache["fonts"].get(key)
        if entry is None or entry.get("fkey") != fkey:
            glyphs = font_charset(path)
            entry = {"fkey": fkey, "coverage": len(glyphs & active) / max(1, len(active)),
                     "covered": "".join(sorted(glyphs & probe))}
            cache["fonts"][key] = entry; dirty = True
        result[key] = {"coverage": entry["coverage"], "covered": entry["covered"]}

    if dirty:
        try:
            manifest_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            logger.warning("coverage manifest not written (%s): %s", manifest_path, e)
    return result


@lru_cache(maxsize=64)
def _load_rgb(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


class RealPaperPool:
    def __init__(self, directory: str | None):
        self.paths = ([str(p) for p in sorted(Path(directory).rglob("*"))
                       if p.suffix.lower() in IMG_EXTS] if directory and Path(directory).exists() else [])

    def __len__(self) -> int:
        return len(self.paths)

    def sample_crop(self, size_wh: tuple[int, int], rng) -> np.ndarray:
        w, h = size_wh
        src = _load_rgb(self.paths[int(rng.integers(0, len(self.paths)))])
        sh, sw = src.shape[:2]
        ar = w / max(1, h)
        cw = min(sw, max(8, int(sh * ar)))
        ch = min(sh, max(8, int(cw / ar)))
        x0 = int(rng.integers(0, max(1, sw - cw + 1)))
        y0 = int(rng.integers(0, max(1, sh - ch + 1)))
        crop = src[y0:y0 + ch, x0:x0 + cw]
        return np.asarray(Image.fromarray(crop).resize((w, h), Image.BILINEAR))
