#!/usr/bin/env python3
"""Download the Cyrillic handwriting dataset (Kaggle) into the project."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.finetune import ensure_cyrillic


def download(root="data/cyrillic"):
    paths = ensure_cyrillic(ROOT / root)
    return {"train_tsv": str(paths.train_tsv), "test_tsv": str(paths.test_tsv) if paths.test_tsv else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/cyrillic")
    print("cyrillic:", download(ap.parse_args().root))


if __name__ == "__main__":
    main()
