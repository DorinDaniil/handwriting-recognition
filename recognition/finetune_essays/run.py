#!/usr/bin/env python3
"""Fine-tune the recognizer on the hand-labelled essays (ideal per-line polygons + text).

Reuses the main stack (model builder, processor, collator, trainer_v2, metrics); only the
data path is new: essay pages -> rectified upright line crops with frame jitter + photometric
augmentation (see dataset.py / augment.py).

    python finetune_essays/run.py --config finetune_essays/config.yaml
    python finetune_essays/run.py --resume --rec-ckpt outputs/other/best
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]        # recognition/
sys.path.insert(0, str(ROOT))

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from finetune_essays.augment import EssayAugmenter
from finetune_essays.dataset import build_essay_datasets
from src.data import TrOCRCollator
from src.finetune.trainer_v2 import train
from src.model import build_processor, build_trocr_small


def _resolve(path):
    local = ROOT / path
    return str(local) if local.exists() else str(path)


def main(config_path, resume, rec_ckpt):
    cfg = OmegaConf.load(config_path)
    if rec_ckpt:
        cfg.model.pretrained = rec_ckpt

    pretrained = _resolve(cfg.model.pretrained)
    tok = cfg.model.get("tokenizer")
    tokenizer_src = _resolve(tok) if tok and (ROOT / tok).exists() else pretrained
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src)

    model, report = build_trocr_small(tokenizer, pretrained, max_length=cfg.model.max_target_len)
    print(report.summary())
    processor = build_processor(tokenizer, pretrained)

    train_ds, test_ds = build_essay_datasets(
        cfg, EssayAugmenter(train=True, p=cfg.get("aug_prob", 0.7)),
        EssayAugmenter(train=False))

    collate = TrOCRCollator(processor, cfg.model.max_target_len)
    bs, nw = cfg.loader.batch_size, cfg.loader.num_workers
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw,
                              collate_fn=collate, pin_memory=True,
                              persistent_workers=nw > 0, drop_last=True)
    val_loader = DataLoader(test_ds, batch_size=max(1, bs // 2), num_workers=min(2, nw),
                            collate_fn=collate)

    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    train(model, processor, train_loader, {"ru": val_loader}, cfg, device, resume=resume)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="finetune_essays/config.yaml")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--rec-ckpt", default=None, help="override model.pretrained")
    args = ap.parse_args()
    main(args.config, args.resume, args.rec_ckpt)
