#!/usr/bin/env python3
"""Evaluate a fine-tuned checkpoint on the test sets, per language and overall.

    python scripts/eval_finetune.py --config configs/finetune.yaml \
        --checkpoint outputs/trocr_small_bi_finetune/best
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data import TrOCRCollator
from src.finetune import Augmenter, HFLineDataset, TsvLineDataset, ensure_cyrillic, load_iam
from src.finetune.metrics import collect_predictions, compute_metrics
from src.model import build_processor, build_trocr_small


def build_test_loaders(cfg, collate):
    eval_aug = Augmenter(train=False)
    bs = max(1, cfg.data.batch_size // 2)
    nw = min(2, cfg.data.num_workers)
    loaders = {}

    cyr = ensure_cyrillic(ROOT / cfg.data.cyrillic_root)
    if cyr.test_tsv:
        ds = TsvLineDataset(cyr.test_tsv, cyr.root, "test", eval_aug)
        loaders["ru"] = DataLoader(ds, batch_size=bs, num_workers=nw, collate_fn=collate)

    iam = load_iam(cfg.data.get("iam"))
    if iam is not None and iam.test is not None:
        ds = HFLineDataset(iam.test, iam.image_key, iam.text_key, eval_aug)
        loaders["en"] = DataLoader(ds, batch_size=bs, num_workers=nw, collate_fn=collate)
    return loaders


def main(config_path, checkpoint, num_beams, max_samples):
    cfg = OmegaConf.load(config_path)
    ckpt = str(ROOT / checkpoint) if (ROOT / checkpoint).exists() else checkpoint

    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model, _ = build_trocr_small(tokenizer, ckpt, max_length=cfg.model.max_target_len)
    processor = build_processor(tokenizer, ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    collate = TrOCRCollator(processor, cfg.model.max_target_len)
    loaders = build_test_loaders(cfg, collate)
    collected = collect_predictions(model, processor, loaders, device,
                                    num_beams=num_beams, max_len=cfg.model.max_target_len,
                                    max_samples=max_samples)

    results = {"checkpoint": checkpoint, "num_beams": num_beams, "max_samples": max_samples}
    all_refs, all_preds = [], []
    for name, (refs, preds) in collected.items():
        m = compute_metrics(refs, preds)
        print(m.row(name))
        results[name] = m.to_dict()
        all_refs += refs
        all_preds += preds
    if len(collected) > 1:
        m = compute_metrics(all_refs, all_preds)
        print(m.row("overall"))
        results["overall"] = m.to_dict()

    out_path = Path(ckpt) / "eval_metrics.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/finetune.yaml")
    ap.add_argument("--checkpoint", default="outputs/trocr_small_bi_finetune/best")
    ap.add_argument("--num-beams", type=int, default=4)
    ap.add_argument("--max-samples", type=int, default=0, help="cap samples per language (0 = full test)")
    args = ap.parse_args()
    main(args.config, args.checkpoint, args.num_beams, args.max_samples)
