"""Visual + throughput sanity check for the synthetic line generator.

Builds a generator, renders a few lines at full difficulty, stacks them vertically
into assets/synth_preview.png (each scaled to a fixed display height so the natural
aspect is visible), prints the ground-truth strings, and times throughput.

    python scripts/demo_synth.py
    python scripts/demo_synth.py --n 10
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from src.synth import HandwrittenLineGenerator, SynthConfig, make_generator  # noqa: E402
from src.synth.config import FontConfig  # noqa: E402


def build_gen() -> HandwrittenLineGenerator:
    cfg = SynthConfig(font=FontConfig(font_dirs=(str(ROOT / "assets" / "fonts"),)))
    return HandwrittenLineGenerator(cfg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="number of preview lines")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--disp-h", type=int, default=64, help="preview row height px")
    ap.add_argument("--max-w", type=int, default=1100, help="preview row width cap px")
    ap.add_argument("--out", type=Path, default=ROOT / "assets" / "synth_preview.png")
    ap.add_argument("--bench", type=int, default=80, help="samples for throughput timing")
    args = ap.parse_args()

    gen = build_gen()
    print(f"FontBank: {len(gen.fonts)} fonts | effects backend: {gen.effects.backend}\n")

    Hd, gap = args.disp_h, 8
    rows = []
    for i in range(args.n):
        img, text = gen.sample(make_generator(args.seed, 0, i), gen.cfg.warmup_steps)
        w, h = img.size
        dw = min(args.max_w, max(1, int(w * Hd / h)))
        rows.append(img.resize((dw, Hd)))
        print(f"{img.size}  {text!r}")

    cw = max(r.width for r in rows)
    canvas = Image.new("RGB", (cw, args.n * (Hd + gap)), (235, 235, 235))
    for i, r in enumerate(rows):
        canvas.paste(r, (0, i * (Hd + gap)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)
    print(f"\nPreview -> {args.out}")

    n = args.bench
    t0 = time.perf_counter()
    for i in range(n):
        gen.sample(make_generator(args.seed + 1, 0, i), gen.cfg.warmup_steps)
    dt = time.perf_counter() - t0
    print(f"Throughput (1 worker): {n/dt:.0f} lines/s  ({1000*dt/n:.2f} ms/line)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
