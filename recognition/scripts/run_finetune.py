#!/usr/bin/env python3
"""Fine-tune the pretrained bilingual TrOCR-small on real handwriting.

Data sources (Cyrillic / IAM / CVL / School Notebooks ...) are selected and configured
under `data.sources` in the config — toggle each with `enabled`.

    python scripts/run_finetune.py --config configs/finetune.yaml
    python scripts/run_finetune.py --resume
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer

from src.data import TrOCRCollator
from src.finetune import Augmenter, build_sources
# from src.finetune.trainer import train
from src.finetune.trainer_v2 import train
from src.model import build_processor, build_trocr_small


def _resolve(path):
    local = ROOT / path
    return str(local) if local.exists() else str(path)


def lang_ratio_sampler(sources, en_ratio):
    totals = {"en": sum(len(s.train) for s in sources if s.lang == "en"),
              "ru": sum(len(s.train) for s in sources if s.lang != "en")}
    weights = []
    for s in sources:
        share = en_ratio if s.lang == "en" else 1.0 - en_ratio
        weights += [share / max(1, totals["en" if s.lang == "en" else "ru"])] * len(s.train)
    return WeightedRandomSampler(weights, num_samples=sum(len(s.train) for s in sources), replacement=True)


def main(config_path, resume):
    cfg = OmegaConf.load(config_path)

    pretrained = _resolve(cfg.model.pretrained)
    tok = cfg.model.get("tokenizer")
    tokenizer_src = _resolve(tok) if tok and (ROOT / tok).exists() else pretrained
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src)

    model, report = build_trocr_small(tokenizer, pretrained, max_length=cfg.model.max_target_len)
    print(report.summary())
    processor = build_processor(tokenizer, pretrained)

    sources = build_sources(cfg.data.sources, ROOT, Augmenter(train=True), Augmenter(train=False))
    if not sources:
        raise SystemExit("no enabled data sources")

    collate = TrOCRCollator(processor, cfg.model.max_target_len)
    nw = cfg.data.num_workers
    train_set = ConcatDataset([s.train for s in sources])
    en_ratio = cfg.data.get("en_ratio")
    langs = {s.lang for s in sources}
    use_sampler = en_ratio is not None and "en" in langs and "ru" in langs
    sampler = lang_ratio_sampler(sources, float(en_ratio)) if use_sampler else None
    train_loader = DataLoader(train_set, batch_size=cfg.data.batch_size, shuffle=sampler is None,
                              sampler=sampler, num_workers=nw, collate_fn=collate, pin_memory=True,
                              persistent_workers=nw > 0, drop_last=True)

    bs, vnw = max(1, cfg.data.batch_size // 2), min(2, nw)
    val_by_lang = defaultdict(list)
    for s in sources:
        if s.test is not None:
            val_by_lang[s.lang].append(s.test)
    val_loaders = {lang: DataLoader(ConcatDataset(tests), batch_size=bs, num_workers=vnw, collate_fn=collate)
                   for lang, tests in val_by_lang.items()}

    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    train(model, processor, train_loader, val_loaders, cfg, device, resume=resume)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/finetune.yaml")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    main(args.config, args.resume)
