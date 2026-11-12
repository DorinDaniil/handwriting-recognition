#!/usr/bin/env python3
"""Evaluate a checkpoint on every held-out set of a config, with both heads.

    python scripts/evaluate.py --checkpoint outputs/finetune/best.pt
    python scripts/evaluate.py --checkpoint outputs/finetune/best.pt --config configs/finetune.yaml --limit 500

Writes eval_metrics.json next to the checkpoint.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from omegaconf import OmegaConf

from htr.model import HTR, ModelConfig
from htr.sources import build_sources
from htr.text import Alphabet
from htr.trainer import evaluate, report


def main(checkpoint, config, limit, device):
    cfg = OmegaConf.load(config)
    saved = torch.load(checkpoint, map_location="cpu")
    cfg.image = saved["image"]
    for spec in cfg.data.sources.values():
        if spec.kind == "blank" or (spec.kind == "synth" and not spec.get("val_samples")):
            spec.enabled = False

    alphabet = Alphabet(saved["alphabet"])
    model = HTR(ModelConfig(**saved["config"]))
    model.load_state_dict(saved["model"])
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    results = evaluate(model.to(device), alphabet, build_sources(cfg, alphabet, ROOT), cfg, device, limit)
    report(results)
    out = Path(checkpoint).with_name("eval_metrics.json")
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="configs/finetune.yaml")
    ap.add_argument("--limit", type=int, default=0, help="lines per set (0 = all)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    main(args.checkpoint, args.config, args.limit, args.device)
