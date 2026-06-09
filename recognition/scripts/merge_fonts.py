"""Copy fonts from your own folders into the project pools (assets/fonts_ru, assets/fonts_en).

Route a mixed source by glyph coverage, or copy explicit RU/EN sources. Files are copied
(not moved), existing names are skipped, nothing is downloaded.

    python scripts/merge_fonts.py --src /my/fonts                  # auto-route by coverage
    python scripts/merge_fonts.py --ru-src /my/ru --en-src /my/en  # explicit
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.synth.assets import font_charset, scan_font_files
from src.synth.config import EN_CHARSET, RU_CHARSET


def copy_into(files, out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    existing = {p.name for p in out.iterdir()}
    n = 0
    for f in files:
        if f.name not in existing:
            shutil.copy2(f, out / f.name)
            existing.add(f.name)
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="*", default=[], help="mixed source dirs (routed by coverage)")
    ap.add_argument("--ru-src", nargs="*", default=[])
    ap.add_argument("--en-src", nargs="*", default=[])
    ap.add_argument("--ru-out", type=Path, default=ROOT / "assets" / "fonts_ru")
    ap.add_argument("--en-out", type=Path, default=ROOT / "assets" / "fonts_en")
    ap.add_argument("--min-coverage", type=float, default=0.80)
    args = ap.parse_args()

    ru = copy_into(scan_font_files(args.ru_src), args.ru_out) if args.ru_src else 0
    en = copy_into(scan_font_files(args.en_src), args.en_out) if args.en_src else 0

    if args.src:
        ru_set, en_set = set(RU_CHARSET), set(EN_CHARSET)
        rfiles, efiles = [], []
        for f in scan_font_files(args.src):
            cov = font_charset(f)
            if len(cov & ru_set) / len(ru_set) >= args.min_coverage:
                rfiles.append(f)
            if len(cov & en_set) / len(en_set) >= args.min_coverage:
                efiles.append(f)
        ru += copy_into(rfiles, args.ru_out)
        en += copy_into(efiles, args.en_out)

    if not (args.src or args.ru_src or args.en_src):
        raise SystemExit("nothing to do: pass --src or --ru-src/--en-src")
    print(f"merged: +{ru} -> {args.ru_out}, +{en} -> {args.en_out}")


if __name__ == "__main__":
    main()
