"""Preview + throughput check: stacks lines into assets/synth_preview.png and times generation."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image

from src.synth import HandwrittenLineGenerator, make_generator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--disp-h", type=int, default=64)
    ap.add_argument("--max-w", type=int, default=1100)
    ap.add_argument("--out", type=Path, default=ROOT / "assets" / "synth_preview.png")
    ap.add_argument("--bench", type=int, default=80)
    args = ap.parse_args()

    gen = HandwrittenLineGenerator.from_dirs(
        ru_text_dirs=[], en_text_dirs=[],
        ru_font_dirs=str(ROOT / "assets" / "fonts_ru"),
        en_font_dirs=str(ROOT / "assets" / "fonts_en"))
    print(f"fonts ru={gen.fonts.n('ru')} en={gen.fonts.n('en')} | effects {gen.effects.backend}\n")

    Hd, gap = args.disp_h, 8
    rows = []
    for i in range(args.n):
        img, text = gen.sample(make_generator(args.seed, 0, i), gen.cfg.warmup_steps)
        dw = min(args.max_w, max(1, int(img.width * Hd / img.height)))
        rows.append(img.resize((dw, Hd)))
        print(f"{img.size}  {text!r}")
    canvas = Image.new("RGB", (max(r.width for r in rows), args.n * (Hd + gap)), (235, 235, 235))
    for i, r in enumerate(rows):
        canvas.paste(r, (0, i * (Hd + gap)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)
    print(f"\nPreview -> {args.out}")

    t0 = time.perf_counter()
    for i in range(args.bench):
        gen.sample(make_generator(args.seed + 1, 0, i), gen.cfg.warmup_steps)
    dt = time.perf_counter() - t0
    print(f"Throughput: {args.bench / dt:.0f} lines/s ({1000 * dt / args.bench:.2f} ms/line)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
