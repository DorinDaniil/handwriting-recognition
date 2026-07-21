"""Essay line dataset: reads the ideal PaddleOCR-style labels.txt, rectifies each line
polygon to an upright crop ON THE FLY from the full page, and (in train) jitters the frame
±`frame_jitter` around the labelled box so the recognizer tolerates loose / tight detector
boxes. Returns {"image": PIL, "text": str} — the same contract as the other line datasets.

labels.txt format (one page per line):
    <folder>/<image>.jpg\t[{"transcription": "...", "points": [[x,y]*4], "score": 1.0}, ...]
"""
from __future__ import annotations

import json
import random
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .geometry import estimate_bg, expand_quad, to_quad, warp_crop


@lru_cache(maxsize=16)
def _load_page(path: str):
    """Decode a page once and cache it with its estimated background colour (per worker)."""
    rgb = np.array(Image.open(path).convert("RGB"))
    return rgb, estimate_bg(rgb)


def parse_labels(labels_path: Path, data_root: Path):
    """-> list of {"page": abs_path, "quad": (4,2) float, "text": str}."""
    records = []
    for line in labels_path.read_text(encoding="utf-8").splitlines():
        if "\t" not in line:
            continue
        rel, payload = line.split("\t", 1)
        page = data_root / rel.strip()
        if not page.exists():
            continue
        try:
            items = json.loads(payload)
        except json.JSONDecodeError:
            continue
        for it in items:
            text = (it.get("transcription") or "").strip()
            pts = it.get("points")
            if text and pts:
                records.append({"page": str(page), "quad": to_quad(pts), "text": text})
    return records


class EssayLineDataset(Dataset):
    def __init__(self, records, augment=None, train=True, jitter_pos=0.02, jitter_neg=0.02,
                 margin_frac=0.06):
        self.records = list(records)
        self.augment = augment
        self.train = train
        self.jpos = jitter_pos          # max outward expansion (+), pulls real pixels
        self.jneg = jitter_neg          # max inward crop (-), tighter than the labelled box
        self.margin_frac = margin_frac

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        rgb, bg = _load_page(rec["page"])
        if self.train and (self.jpos > 0 or self.jneg > 0):
            ew = random.uniform(-self.jneg, self.jpos)   # independent per axis
            eh = random.uniform(-self.jneg, self.jpos)
        else:
            ew, eh = 0.0, 0.0            # eval: exact labelled box
        crop = warp_crop(rgb, expand_quad(rec["quad"], ew, eh), bg, self.margin_frac)
        if crop is None:                       # degenerate box -> fall back to a tiny bg tile
            crop = np.full((8, 8, 3), bg, dtype=np.uint8)
        image = Image.fromarray(crop)
        if self.augment is not None:
            image = self.augment(image)
        return {"image": image, "text": rec["text"]}


def build_essay_datasets(cfg, train_aug, eval_aug):
    """Split by PAGE (no line leakage across train/test), then build the two datasets."""
    data_root = Path(cfg.data_root)
    labels = data_root / cfg.get("labels", "labels.txt")
    records = parse_labels(labels, data_root)
    if not records:
        raise SystemExit(f"no labelled lines found via {labels}")

    pages = sorted({r["page"] for r in records})
    random.Random(cfg.get("seed", 42)).shuffle(pages)
    n_test = int(len(pages) * cfg.get("test_frac", 0.1))
    test_pages = set(pages[:n_test])

    train_recs = [r for r in records if r["page"] not in test_pages]
    test_recs = [r for r in records if r["page"] in test_pages]

    jpos = cfg.get("jitter_pos", 0.02)
    jneg = cfg.get("jitter_neg", 0.02)
    mf = cfg.get("margin_frac", 0.06)
    print(f"essays: pages {len(pages)} (test {len(test_pages)}) | "
          f"lines train {len(train_recs)} test {len(test_recs)}")
    train_ds = EssayLineDataset(train_recs, train_aug, True, jpos, jneg, mf)
    test_ds = EssayLineDataset(test_recs, eval_aug, False, jpos, jneg, mf)
    return train_ds, test_ds
