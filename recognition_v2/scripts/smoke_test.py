#!/usr/bin/env python3
"""A one-minute check that the whole training path works on this machine before a long run.

A small model is trained on a handful of rendered lines through the real stream, trainer,
evaluation and checkpoint code, then loaded back with the inference class. It must read the
lines it has memorised with both heads, and the decoder cache must match a full pass.

    python scripts/smoke_test.py
"""
import multiprocessing as mp
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont

from htr.batching import train_loader
from htr.metrics import score
from htr.model import HTR, ModelConfig
from htr.predictor import Recognizer
from htr.text import BOS, Alphabet
from htr.trainer import train

WORDS = "the quick brown fox jumps over a lazy dog and 42 small birds sing all summer".split()


def render(text: str, size: int) -> Image.Image:
    font = ImageFont.load_default(size)
    width = int(ImageDraw.Draw(Image.new("RGB", (1, 1))).textlength(text, font=font))
    image = Image.new("RGB", (width + 24, size + 28), (245, 242, 232))
    ImageDraw.Draw(image).text((12, 10), text, font=font, fill=(25, 30, 90))
    return image


class Memory:
    """A source that holds its lines in memory; the test set is the training set."""
    name, lang, weight = "memory", "en", 1.0

    def __init__(self, n: int, rng):
        texts = [" ".join(rng.choice(WORDS, int(rng.integers(1, 6)))) for _ in range(n)]
        self.test = [(render(t, int(rng.integers(30, 52))), t) for t in texts]

    def start(self, worker: int):
        pass

    def draw(self, rng, step):
        return self.test[int(rng.integers(len(self.test)))]


def main():
    torch.manual_seed(0)
    alphabet = Alphabet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.create({
        "run": tempfile.mkdtemp(prefix="htr_smoke_"), "seed": 0,
        "image": {"height": 64, "max_width": 768}, "model": {"max_text_len": 64},
        "loss": {"ctc_weight": 0.5, "label_smoothing": 0.1},
        "frame": {"infer": [0.3, 0.2]},
        "data": {"batch_size": 16, "ref_width": 320, "workers": 0, "pool_batches": 2},
        "trainer": {"max_steps": 400, "lr": 2e-3, "weight_decay": 0.0, "warmup_steps": 40, "ema_decay": 0.98,
                    "log_every": 100, "eval_every": 200},
    })
    model = HTR(ModelConfig(vocab=len(alphabet), height=64, dim=128, heads=4, encoder_layers=2,
                            decoder_layers=1, stem=(16, 32, 48, 64, 96), dropout=0.0, max_text_len=64))

    model.eval()
    images, widths = torch.randn(2, 3, 64, 256), torch.tensor([256, 130])
    tokens = torch.randint(3, len(alphabet), (2, 9))
    tokens[:, 0] = BOS
    with torch.no_grad():
        memory, _, valid = model.encoder(images, widths)
        full = model.decoder(tokens, memory, valid)
        cache = model.decoder.new_cache()
        stepwise = torch.cat([model.decoder(tokens[:, i:i + 1], memory, valid, cache) for i in range(9)], 1)
    gap = (full - stepwise).abs().max().item()
    assert gap < 1e-3, f"decoder cache differs from the full pass by {gap}"
    print(f"decoder cache matches the full pass (max difference {gap:.1e})")

    source = Memory(24, np.random.default_rng(0))
    counter = mp.Value("i", 0)
    train(model, alphabet, train_loader([source], alphabet, cfg, counter), [source], cfg, device, counter)

    refs = [text for _, text in source.test]
    for head in ("ctc", "decoder", "auto"):
        reader = Recognizer(Path(cfg.run) / "best.pt", device, head)
        cer = score(refs, reader.recognize([image for image, _ in source.test]))["cer"]
        print(f"{head:8} CER on the memorised lines: {cer:.3f}")
        assert cer < 0.05, f"the {head} head did not memorise 24 lines (CER {cer:.3f})"
    print("ok")


if __name__ == "__main__":
    main()
