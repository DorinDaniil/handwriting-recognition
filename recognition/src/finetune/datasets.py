from __future__ import annotations

import os
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


def _resolve(root: str, subdir: str, name: str) -> str | None:
    base = os.path.basename(name)
    for cand in (os.path.join(root, name), os.path.join(root, base), os.path.join(root, subdir, base)):
        if os.path.exists(cand):
            return cand
    return None


class LineDataset(Dataset):
    """Line images from a list of (image_path, text) pairs."""

    def __init__(self, samples, augment=None):
        self.samples = list(samples)
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, text = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.augment is not None:
            image = self.augment(image)
        return {"image": image, "text": text}


class TsvLineDataset(LineDataset):
    def __init__(self, tsv_path, root, subdir, augment=None):
        samples = []
        for line in Path(tsv_path).read_text(encoding="utf-8").splitlines():
            if "\t" not in line:
                continue
            name, text = line.split("\t", 1)
            text = text.strip()
            path = _resolve(str(root), subdir, name.strip())
            if path and text:
                samples.append((path, text))
        super().__init__(samples, augment)


class HFLineDataset(Dataset):
    def __init__(self, split, image_key="image", text_key="text", augment=None):
        self.split = split
        self.image_key = image_key
        self.text_key = text_key
        self.augment = augment

    def __len__(self):
        return len(self.split)

    def __getitem__(self, idx):
        row = self.split[idx]
        image = row[self.image_key]
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        image = image.convert("RGB")
        if self.augment is not None:
            image = self.augment(image)
        return {"image": image, "text": str(row[self.text_key])}
