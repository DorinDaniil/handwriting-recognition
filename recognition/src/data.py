"""Datasets + collator for TrOCR training on synthetic lines."""
from __future__ import annotations

import multiprocessing as mp

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from .synth import HandwrittenLineGenerator, make_generator


class SynthLineDataset(IterableDataset):
    """Infinite stream. A shared step counter (set by the trainer) drives the curriculum."""
    def __init__(self, gen: HandwrittenLineGenerator, base_seed: int = 42, step_counter=None):
        self.gen, self.base_seed, self.step_counter = gen, base_seed, step_counter

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info is not None else 0
        if info is not None:
            self.gen.warm_cache()
        i = 0
        while True:
            step = self.step_counter.value if self.step_counter is not None else i
            img, text = self.gen.sample(make_generator(self.base_seed, wid, i), step=step)
            yield {"image": img, "text": text}
            i += 1


class FixedSynthValDataset(Dataset):
    """N deterministic samples at full difficulty — a stable proxy val set."""
    def __init__(self, gen: HandwrittenLineGenerator, n: int = 500, seed: int = 123):
        step = gen.cfg.warmup_steps
        self.items = [gen.sample(make_generator(seed, 0, i), step=step) for i in range(n)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img, text = self.items[i]
        return {"image": img, "text": text}


class TrOCRCollator:
    def __init__(self, processor, max_len: int = 64):
        self.p, self.max_len = processor, max_len

    def __call__(self, batch):
        images = [b["image"] for b in batch]
        texts = [b["text"] for b in batch]
        pixel_values = self.p(images=images, return_tensors="pt").pixel_values
        labels = self.p.tokenizer(texts, padding="longest", truncation=True,
                                  max_length=self.max_len, return_tensors="pt").input_ids
        labels[labels == self.p.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels, "texts": texts}


def build_dataloaders(gen, processor, cfg):
    step_counter = mp.Value("i", 0)
    collator = TrOCRCollator(processor, cfg.model.max_target_len)
    nw = cfg.data.num_workers
    train_loader = DataLoader(
        SynthLineDataset(gen, base_seed=cfg.synth.seed, step_counter=step_counter),
        batch_size=cfg.data.batch_size, num_workers=nw, collate_fn=collator,
        pin_memory=True, persistent_workers=nw > 0,
    )
    val_loader = DataLoader(
        FixedSynthValDataset(gen, n=cfg.data.val_samples, seed=cfg.synth.seed + 1),
        batch_size=max(1, cfg.data.batch_size // 2), num_workers=min(2, nw), collate_fn=collator,
    )
    return train_loader, val_loader, step_counter
