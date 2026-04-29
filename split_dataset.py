"""Standalone train/val/test split helper.

    python split_dataset.py --labels labels/labels.txt --out splits --val 0.1 --test 0.05 --seed 42
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from src.dataset import parse_labels_txt


def read_clean_allowlist(paths: list[Path]) -> set[str]:
    allowed: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as file:
            allowed.update(line.strip() for line in file if line.strip())
    return allowed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, default=Path("labels/labels.txt"))
    ap.add_argument("--clean-splits", type=Path, nargs="*", default=None,
                    help="Allowlist split files. Defaults to train_clean.txt and val_clean.txt inside --out if they exist.")
    ap.add_argument("--out", type=Path, default=Path("splits"))
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not (0.0 <= args.val < 1.0):
        raise ValueError("--val must be in [0, 1)")
    if not (0.0 <= args.test < 1.0):
        raise ValueError("--test must be in [0, 1)")
    if args.val + args.test >= 1.0:
        raise ValueError("--val + --test must be < 1")

    items = parse_labels_txt(args.labels)
    total_items = len(items)
    clean_split_paths = args.clean_splits
    if clean_split_paths is None:
        clean_split_paths = [args.out / "train_clean.txt", args.out / "val_clean.txt"]
    clean_allowed = read_clean_allowlist(clean_split_paths)
    if clean_allowed:
        items = [(rel, boxes) for rel, boxes in items if rel in clean_allowed]
        skipped = total_items - len(items)
        print(f"clean allowlist filter: kept={len(items)} skipped_not_clean={skipped}")
        if not items:
            raise RuntimeError(f"No labels matched clean split files: {clean_split_paths}")
    paths = [rel for rel, _ in items]
    rng = random.Random(args.seed)
    rng.shuffle(paths)
    n_test = int(len(paths) * args.test)
    n_val = max(1, int(len(paths) * args.val))
    test = set(paths[:n_test])
    val = set(paths[n_test:n_test + n_val])

    args.out.mkdir(parents=True, exist_ok=True)
    train_path = args.out / "train.txt"
    val_path = args.out / "val.txt"
    test_path = args.out / "test.txt"
    with open(train_path, "w", encoding="utf-8") as f:
        f.writelines(p + "\n" for p in paths if p not in val and p not in test)
    with open(val_path, "w", encoding="utf-8") as f:
        f.writelines(p + "\n" for p in paths if p in val)
    if n_test > 0:
        with open(test_path, "w", encoding="utf-8") as f:
            f.writelines(p + "\n" for p in paths if p in test)
    print(f"train={len(paths) - n_val - n_test}  val={n_val}  test={n_test}")
    print(f"wrote {train_path}")
    print(f"wrote {val_path}")
    if n_test > 0:
        print(f"wrote {test_path}")


if __name__ == "__main__":
    main()
