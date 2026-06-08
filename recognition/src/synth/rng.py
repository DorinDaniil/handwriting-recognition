"""Worker-safe randomness + curriculum helpers.

The detection ``Augmenter`` uses the global ``np.random`` state. That is fine for
a finite map-style ``Dataset``, but an *infinite* ``IterableDataset`` forked
across ``num_workers`` would then produce correlated / duplicated samples in
every worker. So every component in this package draws from an explicit
:class:`numpy.random.Generator` created from a ``SeedSequence`` of
``(base_seed, worker_id, draw_index)`` — independent streams, fully reproducible.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def make_generator(base_seed: int, worker_id: int = 0, draw_index: int = 0) -> np.random.Generator:
    """Independent, reproducible RNG for one sample drawn by one worker."""
    ss = np.random.SeedSequence([int(base_seed), int(worker_id), int(draw_index)])
    return np.random.default_rng(ss)


# --- curriculum easing ----------------------------------------------------

def lerp(lo: float, hi: float, t: float) -> float:
    """Linear interpolate lo→hi as t goes 0→1."""
    return lo + (hi - lo) * float(t)


def scale_p(p: float, t: float, floor: float = 0.2) -> float:
    """Scale a probability by curriculum t, keeping a floor so early steps still
    see *some* variety: at t=0 returns ``floor*p``, at t=1 returns ``p``."""
    return (floor + (1.0 - floor) * float(t)) * p


# --- small sampling helpers (all take an explicit Generator) --------------

def chance(rng: np.random.Generator, p: float) -> bool:
    return bool(rng.random() < p)


def uniform(rng: np.random.Generator, pair: Sequence[float]) -> float:
    lo, hi = pair
    return float(rng.uniform(lo, hi))


def randint(rng: np.random.Generator, pair: Sequence[int]) -> int:
    """Inclusive integer in [pair[0], pair[1]]."""
    lo, hi = int(pair[0]), int(pair[1])
    return int(rng.integers(lo, hi + 1)) if hi >= lo else lo


def eased_uniform(rng: np.random.Generator, pair: Sequence[float], t: float) -> float:
    """Uniform sample whose *upper* bound grows with the curriculum: at t=0 it
    sits near ``pair[0]`` (easy), at t=1 it spans the full ``(lo, hi)`` range."""
    lo, hi = pair
    hi_t = lerp(lo, hi, t)
    return float(rng.uniform(lo, hi_t))


def choice(rng: np.random.Generator, seq, weights=None):
    """Pick one element of ``seq`` (optionally weighted)."""
    n = len(seq)
    if weights is None:
        return seq[int(rng.integers(0, n))]
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    return seq[int(rng.choice(n, p=w))]
