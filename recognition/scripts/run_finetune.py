#!/usr/bin/env python3
"""Fine-tune the pretrained bilingual TrOCR-small on real handwriting
(Cyrillic Kaggle dataset + IAM English).

    python scripts/run_finetune.py --config configs/finetune.yaml
    python scripts/run_finetune.py --resume
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset, DataLoader
from transformers import AutoTokenizer

from src.data import TrOCRCollator
from src.finetune import Augmenter, HFLineDataset, TsvLineDataset, ensure_cyrillic, load_iam
from src.model import build_processor, build_trocr_small
from src.train import train_model


def _resolve(path):
    local = ROOT / path
    return str(local) if local.exists() else str(path)


def infinite(loader):
    while True:
        yield from loader


def build_datasets(cfg, train_aug, eval_aug):
    cyr = ensure_cyrillic(_resolve(cfg.data.cyrillic_root))
    train_sets = [TsvLineDataset(cyr.train_tsv, cyr.root, "train", train_aug)]
    val_sets = [TsvLineDataset(cyr.test_tsv, cyr.root, "test", eval_aug)] if cyr.test_tsv else []

    iam = load_iam(cfg.data.get("iam"))
    if iam is not None:
        train_sets.append(HFLineDataset(iam.train, iam.image_key, iam.text_key, train_aug))
        if iam.test is not None:
            val_sets.append(HFLineDataset(iam.test, iam.image_key, iam.text_key, eval_aug))
    return ConcatDataset(train_sets), ConcatDataset(val_sets)


def main(config_path, resume):
    cfg = OmegaConf.load(config_path)

    pretrained = _resolve(cfg.model.pretrained)
    tok = cfg.model.get("tokenizer")
    tokenizer_src = _resolve(tok) if tok and (ROOT / tok).exists() else pretrained
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src)

    model, report = build_trocr_small(tokenizer, pretrained, max_length=cfg.model.max_target_len)
    print(report.summary())
    processor = build_processor(tokenizer, pretrained)

    train_set, val_set = build_datasets(cfg, Augmenter(train=True), Augmenter(train=False))
    print(f"train {len(train_set)} lines | val {len(val_set)} lines")

    collate = TrOCRCollator(processor, cfg.model.max_target_len)
    nw = cfg.data.num_workers
    train_loader = DataLoader(train_set, batch_size=cfg.data.batch_size, shuffle=True,
                              num_workers=nw, collate_fn=collate, pin_memory=True,
                              persistent_workers=nw > 0, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=max(1, cfg.data.batch_size // 2),
                            num_workers=min(2, nw), collate_fn=collate)

    train_model(model, processor, infinite(train_loader), val_loader, cfg, resume=resume)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/finetune.yaml")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    main(args.config, args.resume)
