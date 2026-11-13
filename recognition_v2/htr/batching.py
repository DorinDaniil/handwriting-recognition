"""Batches of lines with similar width.

Lines are resized to one height and keep their own width, so a batch is padded to its widest
line. Samples are drawn into a pool, sorted by width and cut into batches under a fixed pixel
budget: wide lines make smaller batches, narrow ones larger, and little is spent on padding.
"""
from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info

from .image import fit_height, to_batch
from .model import STRIDE
from .text import BOS, EOS, Alphabet


def collate(items) -> dict:
    arrays, ids, texts = zip(*items)
    images, widths = to_batch(list(arrays))
    longest = max(len(i) for i in ids)
    targets = torch.zeros(len(ids), max(longest, 1), dtype=torch.long)
    decoder_in = torch.full((len(ids), longest + 1), EOS, dtype=torch.long)
    decoder_out = torch.full((len(ids), longest + 1), -100, dtype=torch.long)
    decoder_in[:, 0] = BOS
    for row, seq in enumerate(ids):
        seq = torch.tensor(seq, dtype=torch.long)
        targets[row, :len(seq)] = seq
        decoder_in[row, 1:len(seq) + 1] = seq
        decoder_out[row, :len(seq)] = seq
        decoder_out[row, len(seq)] = EOS
    return {"images": images, "widths": widths, "targets": targets,
            "target_lengths": torch.tensor([len(i) for i in ids]),
            "decoder_in": decoder_in, "decoder_out": decoder_out, "texts": list(texts)}


class TrainStream(IterableDataset):
    """Endless stream of training batches drawn from the sources by their weights."""

    def __init__(self, sources, alphabet: Alphabet, cfg, counter):
        self.sources = [s for s in sources if s.weight > 0]
        self.alphabet, self.counter = alphabet, counter
        self.height, self.max_width = cfg.image.height, cfg.image.max_width
        self.max_text = cfg.model.get("max_text_len", 160) - 1
        self.budget = cfg.data.batch_size * cfg.data.ref_width
        self.max_batch = 2 * cfg.data.batch_size
        self.pool = cfg.data.get("pool_batches", 6) * cfg.data.batch_size
        self.seed = cfg.get("seed", 42)

    def encode(self, sample):
        """(image, text) -> (uint8 image at model height, character ids, text), or None to skip:
        a label that cannot be trusted, or more characters than the CTC head has frames for."""
        if sample is None or sample[0] is None:
            return None
        image, text = sample
        text = self.alphabet.clean(text)
        if text is None or len(text) > self.max_text:
            return None
        array = fit_height(image, self.height, self.max_width)
        ids = self.alphabet.encode(text)
        repeats = sum(a == b for a, b in zip(ids, ids[1:]))
        if len(ids) + repeats > -(-array.shape[1] // STRIDE):
            return None
        return array, ids, text

    def cut(self, pool):
        """Width-sorted pool -> (batch, is_full). Only the last batch can be under-filled."""
        batch = []
        for item in pool:
            width = item[0].shape[1]
            if batch and ((len(batch) + 1) * width > self.budget or len(batch) == self.max_batch):
                yield batch, True
                batch = []
            batch.append(item)
        if batch:
            yield batch, 2 * len(batch) * batch[-1][0].shape[1] >= self.budget

    def __iter__(self):
        info = get_worker_info()
        worker = info.id if info is not None else 0
        rng = np.random.default_rng([self.seed, worker])
        for source in self.sources:
            source.start(worker)
        weights = np.array([s.weight for s in self.sources], dtype=np.float64)
        weights /= weights.sum()

        pool = []
        while True:
            while len(pool) < self.pool:
                source = self.sources[int(rng.choice(len(self.sources), p=weights))]
                item = self.encode(source.draw(rng, self.counter.value))
                if item is not None:
                    pool.append(item)
            pool.sort(key=lambda item: item[0].shape[1])
            batches = list(self.cut(pool))
            pool = [] if batches[-1][1] else batches.pop()[0]
            for i in rng.permutation(len(batches)):
                yield collate(batches[i][0])


def train_loader(sources, alphabet, cfg, counter) -> DataLoader:
    workers = cfg.data.workers
    return DataLoader(TrainStream(sources, alphabet, cfg, counter), batch_size=None, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0)


class TestLines(Dataset):
    """A held-out list of (image, text), prepared exactly as at inference."""

    def __init__(self, items, cfg):
        self.items, self.height, self.max_width = items, cfg.image.height, cfg.image.max_width

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        image, text = self.items[i]
        return (fit_height(image, self.height, self.max_width), text) if image is not None else None


def _by_width(items):
    return sorted((item for item in items if item is not None), key=lambda item: item[0].shape[1])


def test_batches(items, cfg, limit: int = 0):
    """Yield (images, widths, texts) over a test list, lines of similar width together.
    With a limit, a random sample of the list is read — the same lines at every evaluation."""
    size = cfg.data.get("eval_batch_size", cfg.data.batch_size)
    data = TestLines(items, cfg)
    if limit and limit < len(data):
        data = torch.utils.data.Subset(data, sorted(random.Random(0).sample(range(len(data)), limit)))
    loader = DataLoader(data, batch_size=8 * size, num_workers=min(4, cfg.data.workers), collate_fn=_by_width)
    for chunk in loader:
        for i in range(0, len(chunk), size):
            arrays, texts = zip(*chunk[i:i + size])
            yield (*to_batch(list(arrays)), list(texts))
