#!/usr/bin/env python3
"""Evaluate a fine-tuned TrOCR checkpoint on the test set of every enabled source.

Each source's test set is scored independently (its own CER/WER/NES, labelled with the
source language) — no averaging or combined score. Results are printed and saved to
<checkpoint>/eval_metrics.json incrementally as each source finishes.

    python scripts/eval_finetune.py --config configs/finetune.yaml \
        --checkpoint outputs/trocr_small_bi_finetune/best
    python scripts/eval_finetune.py --num-beams 1 --max-samples 300   # quick check
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
from src.finetune import Augmenter, build_sources
from src.finetune.metrics import collect_predictions, compute_metrics
from src.model import build_processor, build_trocr_small


def main(config_path, checkpoint, num_beams, max_samples):
    cfg = OmegaConf.load(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = str(ROOT / checkpoint) if (ROOT / checkpoint).exists() else checkpoint

    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model, _ = build_trocr_small(tokenizer, ckpt, max_length=cfg.model.max_target_len)
    processor = build_processor(tokenizer, ckpt)
    model.to(device)

    sources = build_sources(cfg.data.sources, ROOT, Augmenter(train=False), Augmenter(train=False))
    collate = TrOCRCollator(processor, cfg.model.max_target_len)
    bs, nw = max(1, cfg.data.batch_size // 2), min(2, cfg.data.num_workers)

    out_path = Path(ckpt) / "eval_metrics.json"
    results = {"checkpoint": checkpoint, "num_beams": num_beams, "max_samples": max_samples}
    for s in sources:
        if s.test is None or len(s.test) == 0:
            continue
        n = min(len(s.test), max_samples) if max_samples else len(s.test)
        print(f"evaluating {s.name} [{s.lang}] ({n} lines)...")
        loader = DataLoader(s.test, batch_size=bs, num_workers=nw, collate_fn=collate)
        (refs, preds), = collect_predictions(model, processor, {s.name: loader}, device,
                                             num_beams=num_beams, max_len=cfg.model.max_target_len,
                                             max_samples=max_samples).values()
        m = compute_metrics(refs, preds)
        print(f"[{s.lang}] {m.row(s.name)}")
        results[s.name] = {"lang": s.lang, **m.to_dict()}
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/finetune.yaml")
    ap.add_argument("--checkpoint", default="outputs/trocr_small_bi_finetune/best")
    ap.add_argument("--num-beams", type=int, default=1)
    ap.add_argument("--max-samples", type=int, default=0, help="cap samples per source (0 = full test)")
    args = ap.parse_args()
    main(args.config, args.checkpoint, args.num_beams, args.max_samples)
