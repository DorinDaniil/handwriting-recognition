from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

CYRILLIC_DATASET = "constantinwerner/cyrillic-handwriting-dataset"


@dataclass
class CyrillicPaths:
    root: Path
    train_tsv: Path
    test_tsv: Path


@dataclass
class IamData:
    train: object
    test: object
    image_key: str
    text_key: str


def _find(base, name):
    hits = list(Path(base).rglob(name))
    return hits[0] if hits else None


def _copy_into(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def ensure_cyrillic(root) -> CyrillicPaths:
    root = Path(root)
    if _find(root, "train.tsv") is None:
        import kagglehub
        _copy_into(Path(kagglehub.dataset_download(CYRILLIC_DATASET)), root)
    train_tsv = _find(root, "train.tsv")
    test_tsv = _find(root, "test.tsv")
    if train_tsv is None:
        raise FileNotFoundError(f"train.tsv not found under {root}")
    return CyrillicPaths(train_tsv.parent, train_tsv, test_tsv)


def load_iam(cfg) -> IamData | None:
    if not cfg or not cfg.get("enabled", False):
        return None
    from datasets import concatenate_datasets, load_dataset
    ds = load_dataset(cfg.get("hf_id", "Teklia/IAM-line"), cache_dir=cfg.get("cache_dir"))
    train_split = cfg.get("train_split", "train")
    test_split = cfg.get("test_split", "test")

    parts = [ds[train_split]]
    for name in ds:                                  # fold any validation split into train
        if name not in (train_split, test_split) and "val" in name.lower():
            parts.append(ds[name])
    train = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    return IamData(
        train=train,
        test=ds[test_split] if test_split in ds else None,
        image_key=cfg.get("image_key", "image"),
        text_key=cfg.get("text_key", "text"),
    )
