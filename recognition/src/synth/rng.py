"""Per-worker RNG + curriculum easing helpers."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def make_generator(base_seed: int, worker_id: int = 0, draw_index: int = 0) -> np.random.Generator:
    """Independent reproducible RNG per (worker, draw) — keeps DataLoader workers decorrelated."""
    return np.random.default_rng(np.random.SeedSequence([int(base_seed), int(worker_id), int(draw_index)]))


def lerp(lo: float, hi: float, t: float) -> float:
    return lo + (hi - lo) * float(t)


def scale_p(p: float, t: float, floor: float = 0.2) -> float:
    return (floor + (1.0 - floor) * float(t)) * p


def chance(rng, p: float) -> bool:
    return bool(rng.random() < p)


def uniform(rng, pair: Sequence[float]) -> float:
    return float(rng.uniform(pair[0], pair[1]))


def randint(rng, pair: Sequence[int]) -> int:
    lo, hi = int(pair[0]), int(pair[1])
    return int(rng.integers(lo, hi + 1)) if hi >= lo else lo


def eased_uniform(rng, pair: Sequence[float], t: float) -> float:
    """One-sided range whose upper bound grows with t (lo stays neutral)."""
    return float(rng.uniform(pair[0], lerp(pair[0], pair[1], t)))


def eased_centered(rng, pair: Sequence[float], t: float) -> float:
    """Range that shrinks toward 0 as t->0 (for slant / rotation / jitter)."""
    return float(rng.uniform(pair[0] * t, pair[1] * t)) if t > 0 else 0.0


def choice(rng, seq, weights=None):
    if weights is None:
        return seq[int(rng.integers(0, len(seq)))]
    w = np.asarray(weights, dtype=np.float64)
    return seq[int(rng.choice(len(seq), p=w / w.sum()))]
