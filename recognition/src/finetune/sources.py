"""Dataset registry: one builder per data source, selected/configured from `data.sources`.

Each enabled source becomes a Source(name, lang, train, test) of torch datasets
(TsvLineDataset / HFLineDataset). Kinds:
  cyrillic  -> Kaggle Cyrillic (downloads if missing)
  iam       -> HuggingFace line dataset
  tsv       -> a local manifest dir (CVL, School Notebooks, or any *.tsv crop set)
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from .datasets import HFLineDataset, LineDataset, TsvLineDataset
from .download import ensure_cyrillic, load_iam


@dataclass
class Source:
    name: str
    lang: str
    train: object
    test: object | None


def _cyrillic(spec, root, train_aug, eval_aug):
    paths = ensure_cyrillic(root / spec.get("root", "data/cyrillic"))
    train = TsvLineDataset(paths.train_tsv, paths.root, "train", train_aug)
    test = TsvLineDataset(paths.test_tsv, paths.root, "test", eval_aug) if paths.test_tsv else None
    return train, test


def _iam(spec, root, train_aug, eval_aug):
    iam = load_iam(spec)
    if iam is None:
        return None, None
    train = HFLineDataset(iam.train, iam.image_key, iam.text_key, train_aug)
    test = HFLineDataset(iam.test, iam.image_key, iam.text_key, eval_aug) if iam.test else None
    return train, test


def _tsv(spec, root, train_aug, eval_aug):
    base = root / spec["root"]
    train_tsv = next(base.glob(spec.get("train_glob", "train.tsv")), None)
    if train_tsv is None:
        return None, None
    subdir = spec.get("subdir", "")
    train = TsvLineDataset(train_tsv, train_tsv.parent, subdir, train_aug)
    test = None
    if spec.get("test_glob"):
        test_tsv = next(base.glob(spec["test_glob"]), None)
        if test_tsv is not None:
            test = TsvLineDataset(test_tsv, test_tsv.parent, subdir, eval_aug)
    return train, test


def _pairs(spec, root, train_aug, eval_aug):
    images = Path(spec["images"])
    texts = Path(spec["texts"])
    pairs = []
    for img in sorted(images.glob("*.png")):
        txt = texts / (img.stem + ".txt")
        if txt.exists():
            text = txt.read_text(encoding="utf-8").strip()
            if text:
                pairs.append((str(img), text))
    if not pairs:
        return None, None
    random.Random(spec.get("seed", 42)).shuffle(pairs)
    n_test = int(len(pairs) * spec.get("test_frac", 0.1))
    return LineDataset(pairs[n_test:], train_aug), LineDataset(pairs[:n_test], eval_aug)


_BUILDERS = {"cyrillic": _cyrillic, "iam": _iam, "tsv": _tsv, "pairs": _pairs}


def build_sources(sources_cfg, root, train_aug, eval_aug):
    root = Path(root)
    out = []
    for name, spec in dict(sources_cfg).items():
        if not spec.get("enabled", True):
            continue
        builder = _BUILDERS.get(spec.get("kind"))
        if builder is None:
            raise ValueError(f"source '{name}': unknown kind {spec.get('kind')!r}")
        train, test = builder(spec, root, train_aug, eval_aug)
        if train is None:
            print(f"source '{name}': no data found (skipped)")
            continue
        lang = spec.get("lang", "ru")
        out.append(Source(name, lang, train, test))
        print(f"source '{name}' [{lang}]: train={len(train)} test={len(test) if test else 0}")
    return out
