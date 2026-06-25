#!/usr/bin/env python3
"""Download the IAM lines dataset (HuggingFace) into the local cache."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.finetune import load_iam


def download(iam_cfg):
    iam = load_iam(iam_cfg)
    if iam is None:
        return {"status": "skipped (disabled or unavailable)"}
    return {"train": len(iam.train), "test": len(iam.test) if iam.test else 0}


def main():
    from omegaconf import OmegaConf
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/finetune.yaml")
    cfg = OmegaConf.load(ROOT / ap.parse_args().config)
    print("iam:", download(cfg.data.get("iam")))


if __name__ == "__main__":
    main()
