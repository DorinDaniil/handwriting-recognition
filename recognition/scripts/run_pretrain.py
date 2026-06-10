#!/usr/bin/env python3
"""Pretrain bilingual TrOCR-small on synthetic lines.

    python scripts/run_pretrain.py --config configs/pretrain_small.yaml
    python scripts/run_pretrain.py --resume
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from omegaconf import OmegaConf
from transformers import AutoTokenizer

from src.synth import HandwrittenLineGenerator
from src.model import build_trocr_small, build_processor
from src.data import build_dataloaders
from src.train import train_model


def main(config_path, resume):
    cfg = OmegaConf.load(config_path)

    tok = cfg.model.get("tokenizer")
    have_tok = tok and (ROOT / tok).exists()
    if not have_tok:
        print(f"tokenizer '{tok}' not found -> using {cfg.model.pretrained}")
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / tok) if have_tok else cfg.model.pretrained)

    model, report = build_trocr_small(tokenizer, cfg.model.pretrained, max_length=cfg.model.max_target_len)
    print(report.summary())
    
    processor = build_processor(tokenizer, cfg.model.pretrained)
    print('processor ready')

    gen = HandwrittenLineGenerator.from_dirs(
        ru_text_dirs=list(cfg.data.ru_text_dirs), en_text_dirs=list(cfg.data.en_text_dirs),
        ru_font_dirs=list(cfg.data.ru_font_dirs), en_font_dirs=list(cfg.data.en_font_dirs),
        ru_text_weights=list(cfg.data.get("ru_text_weights", [])),
        en_text_weights=list(cfg.data.get("en_text_weights", [])),
        p_ru=cfg.data.p_ru, len_chars=tuple(cfg.data.len_chars), p_hyphenate=cfg.data.p_hyphenate,
        warmup_steps=cfg.synth.warmup_steps, seed=cfg.synth.seed,
        curriculum=bool(cfg.synth.get("curriculum", True)),
        synth_overrides=cfg.synth)            # stage-2 sub-blocks (corpus/render/effects/neighbors); no-op for stage-1
    print('gen ready')

    train_loader, val_loader, step_counter = build_dataloaders(gen, processor, cfg)
    train_model(model, processor, train_loader, val_loader, cfg, step_counter=step_counter, resume=resume)

    print('loaders ready')


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pretrain_small.yaml")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    main(args.config, args.resume)
