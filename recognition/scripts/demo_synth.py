"""Visual + throughput sanity check for the synthetic line generator.

Builds a generator, renders a grid sweeping the curriculum difficulty (rows) and
saves it to assets/synth_preview.png, prints the ground-truth strings, and times
single-worker throughput.

    python scripts/demo_synth.py                 # default 6 samples per t row
    python scripts/demo_synth.py --n 8 --cell 256
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
    ap.add_argument("--n", type=int, default=6, help="samples per curriculum row")
    ap.add_argument("--cell", type=int, default=256, help="preview cell size px")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=ROOT / "assets" / "synth_preview.png")
    ap.add_argument("--bench", type=int, default=100, help="samples for throughput timing")
    args = ap.parse_args()

    gen = build_gen()
    print(f"FontBank: {len(gen.fonts)} fonts | effects backend: {gen.effects.backend}\n")

    ts = [0.0, 0.34, 0.67, 1.0]
    warmup = max(1, gen.cfg.warmup_steps)
    cell = args.cell
    grid = Image.new("RGB", (args.n * cell, len(ts) * cell), (235, 235, 235))

    draw_i = 0
    for r, t in enumerate(ts):
        step = int(t * warmup)
        texts = []
        for c in range(args.n):
            rng = make_generator(args.seed, 0, draw_i); draw_i += 1
            img, text = gen.sample(rng, step)
            grid.paste(img.resize((cell, cell)), (c * cell, r * cell))
            texts.append(text)
        print(f"t={t:.2f}: " + " | ".join(repr(x) for x in texts))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    grid.save(args.out)
    print(f"\nPreview grid -> {args.out}")

    # throughput (natural lines, no letterbox overhead distinction is negligible)
    n = args.bench
    t0 = time.perf_counter()
    for i in range(n):
        rng = make_generator(args.seed + 1, 0, i)
        gen.sample(rng, warmup)
    dt = time.perf_counter() - t0
    print(f"Throughput (1 worker): {n/dt:.0f} lines/s  ({1000*dt/n:.2f} ms/line)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
