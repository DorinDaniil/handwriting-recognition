#!/usr/bin/env python3
"""Pretrain TrOCR-small (Russian) on synthetic lines.

    python scripts/run_pretrain.py --config configs/pretrain_small.yaml
    python scripts/run_pretrain.py --resume
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from omegaconf import OmegaConf            # noqa: E402
from transformers import AutoTokenizer     # noqa: E402

from src.synth import HandwrittenLineGenerator   # noqa: E402
from src.model import build_trocr_small, build_processor  # noqa: E402
from src.data import build_dataloaders     # noqa: E402
from src.train import train_model          # noqa: E402


def main(config_path: str, resume: bool):
    cfg = OmegaConf.load(config_path)

    tok = cfg.model.get("tokenizer")
    tok_path = tok if tok and (ROOT / tok).exists() else cfg.model.pretrained
    if tok_path != tok:
        print(f"tokenizer '{tok}' not found -> using {cfg.model.pretrained} (train one: scripts/train_tokenizer.py)")
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / tok) if tok_path == tok else cfg.model.pretrained)

    model, report = build_trocr_small(tokenizer, cfg.model.pretrained, max_length=cfg.model.max_target_len)
    print(report.summary())
    processor = build_processor(tokenizer, cfg.model.pretrained)

    gen = HandwrittenLineGenerator.from_dirs(
        text_dirs=list(cfg.data.text_dirs),
        font_dirs=list(cfg.data.font_dirs),
        len_chars=tuple(cfg.data.len_chars),
        p_hyphenate=cfg.data.p_hyphenate,
        p_words=cfg.data.p_words,
        p_random=cfg.data.p_random,
        warmup_steps=cfg.synth.warmup_steps,
        seed=cfg.synth.seed,
    )
    train_loader, val_loader, step_counter = build_dataloaders(gen, processor, cfg)
    train_model(model, processor, train_loader, val_loader, cfg,
                step_counter=step_counter, resume=resume)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pretrain TrOCR-small (RU) on synthetic lines.")
    ap.add_argument("--config", default="configs/pretrain_small.yaml")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    main(args.config, args.resume)
