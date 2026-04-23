"""Standalone train/val split helper.

    python split_dataset.py --labels labels/labels.txt --out splits --val 0.1 --seed 42
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from src.dataset import parse_labels_txt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, default=Path("labels/labels.txt"))
    ap.add_argument("--out", type=Path, default=Path("splits"))
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    items = parse_labels_txt(args.labels)
    paths = [rel for rel, _ in items]
    rng = random.Random(args.seed)
    rng.shuffle(paths)
    n_val = max(1, int(len(paths) * args.val))
    val = set(paths[:n_val])

    args.out.mkdir(parents=True, exist_ok=True)
    train_path = args.out / "train.txt"
    val_path = args.out / "val.txt"
    with open(train_path, "w", encoding="utf-8") as f:
        f.writelines(p + "\n" for p in paths if p not in val)
    with open(val_path, "w", encoding="utf-8") as f:
        f.writelines(p + "\n" for p in paths if p in val)
    print(f"train={len(paths) - n_val}  val={n_val}")
    print(f"wrote {train_path}")
    print(f"wrote {val_path}")


if __name__ == "__main__":
    main()
