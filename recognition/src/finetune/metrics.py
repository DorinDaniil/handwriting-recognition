from __future__ import annotations

from dataclasses import dataclass

import jiwer
import torch
from rapidfuzz.distance import Levenshtein


@dataclass
class Metrics:
    samples: int
    cer: float
    wer: float
    nes_char: float
    nes_word: float

    def row(self, name: str) -> str:
        return (f"{name:<8} n={self.samples:<6} CER={self.cer:.4f} WER={self.wer:.4f} "
                f"NES_char={self.nes_char:.4f} NES_word={self.nes_word:.4f}")

    def to_dict(self) -> dict:
        return {"samples": self.samples, "cer": self.cer, "wer": self.wer,
                "nes_char": self.nes_char, "nes_word": self.nes_word}


def compute_metrics(refs, preds) -> Metrics:
    n = len(refs)
    if n == 0:
        return Metrics(0, 0.0, 0.0, 1.0, 1.0)
    nes_char = sum(Levenshtein.normalized_similarity(r, h) for r, h in zip(refs, preds)) / n
    nes_word = sum(Levenshtein.normalized_similarity(r.split(), h.split()) for r, h in zip(refs, preds)) / n
    return Metrics(n, float(jiwer.cer(refs, preds)), float(jiwer.wer(refs, preds)), nes_char, nes_word)


@torch.no_grad()
def collect_predictions(model, processor, loaders, device, num_beams=1, max_len=128, max_samples=0):
    model.eval()
    out = {}
    for name, loader in loaders.items():
        refs, preds = [], []
        for batch in loader:
            ids = model.generate(batch["pixel_values"].to(device), num_beams=num_beams, max_length=max_len)
            preds += processor.tokenizer.batch_decode(ids, skip_special_tokens=True)
            refs += batch["texts"]
            if max_samples and len(refs) >= max_samples:
                break
        out[name] = (refs[:max_samples] if max_samples else refs,
                     preds[:max_samples] if max_samples else preds)
    return out
