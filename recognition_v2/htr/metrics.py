"""Recognition metrics over (reference, hypothesis) pairs."""
from __future__ import annotations

import jiwer
from rapidfuzz.distance import Levenshtein


def score(refs: list[str], hyps: list[str]) -> dict:
    """CER / WER over the whole set; NES = mean per-line normalised edit similarity."""
    n = len(refs)
    if n == 0:
        return {"samples": 0, "cer": 0.0, "wer": 0.0, "nes_char": 1.0, "nes_word": 1.0}
    return {
        "samples": n,
        "cer": float(jiwer.cer(refs, hyps)),
        "wer": float(jiwer.wer(refs, hyps)),
        "nes_char": sum(Levenshtein.normalized_similarity(r, h) for r, h in zip(refs, hyps)) / n,
        "nes_word": sum(Levenshtein.normalized_similarity(r.split(), h.split()) for r, h in zip(refs, hyps)) / n,
    }


def row(name: str, m: dict) -> str:
    return (f"{name:<28} n={m['samples']:<6} CER={m['cer']:.4f} WER={m['wer']:.4f} "
            f"NES_char={m['nes_char']:.4f} NES_word={m['nes_word']:.4f}")


def harmonic(values) -> float:
    values = list(values)
    if not values or any(v <= 0 for v in values):
        return 0.0
    return len(values) / sum(1.0 / v for v in values)
