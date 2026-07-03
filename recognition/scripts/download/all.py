#!/usr/bin/env python3
"""Download all fine-tuning datasets in one go.

    python scripts/download/all.py                              # all datasets
    python scripts/download/all.py --datasets cyrillic iam      # a subset
    python scripts/download/all.py --preview                    # parse a few samples, no write

Per-dataset roots/options come from the config (configs/finetune.yaml); a failure in one
dataset is reported and skipped, the rest still run.
"""
import argparse
import importlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from omegaconf import OmegaConf

DATASETS = {
    "cyrillic": lambda cfg, pv: ("cyrillic", {"root": cfg.data.get("cyrillic_root", "data/cyrillic")}),
    "iam": lambda cfg, pv: ("iam", {"iam_cfg": cfg.data.get("iam")}),
    "cvl": lambda cfg, pv: ("cvl", {"root": cfg.data.get("cvl_root", "data/cvl"), "preview": pv}),
    "imgur5k": lambda cfg, pv: ("imgur5k", {
        "out": cfg.data.sources.get("imgur5k", {}).get("root", "data/imgur5k"),
        "split": "all", "preview": pv}),   # clones + downloads + crops + cleans up by itself
    "school_notebooks_ru": lambda cfg, pv: ("school_notebooks",
        {"root": "data/school_notebooks_ru", "repo": "ai-forever/school_notebooks_RU", "preview": pv}),
    "school_notebooks_en": lambda cfg, pv: ("school_notebooks",
        {"root": "data/school_notebooks_en", "repo": "ai-forever/school_notebooks_EN", "preview": pv}),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/finetune.yaml")
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()
    cfg = OmegaConf.load(ROOT / args.config)

    for name in args.datasets:
        print(f"== {name} ==")
        try:
            module_name, kwargs = DATASETS[name](cfg, args.preview)
            module = importlib.import_module(module_name)
            print("  ", module.download(**kwargs))
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
