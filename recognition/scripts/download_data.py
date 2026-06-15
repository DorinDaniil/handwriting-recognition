#!/usr/bin/env python3
"""Download the fine-tuning datasets into the project (Cyrillic + IAM).

    python scripts/download_data.py --config configs/finetune.yaml
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from omegaconf import OmegaConf

from src.finetune import ensure_cyrillic, load_iam


def main(config_path):
    cfg = OmegaConf.load(config_path)

    cyr = ensure_cyrillic(ROOT / cfg.data.cyrillic_root)
    print(f"cyrillic -> {cyr.root}")
    print(f"  train: {cyr.train_tsv}")
    print(f"  test:  {cyr.test_tsv}")

    iam = load_iam(cfg.data.get("iam"))
    if iam is None:
        print("iam -> skipped")
    else:
        print(f"iam -> train {len(iam.train)} | test {len(iam.test) if iam.test else 0}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/finetune.yaml")
    main(ap.parse_args().config)
