#!/usr/bin/env python3
"""Score hwr200 (image, label) pairs with the trained TrOCR model and split them into
good / bad by a CER percentile threshold. Nothing is deleted — good and bad pairs are
written to separate JSON files per split (train/test), for review and selective use.

    python scripts/filter_hwr200.py --checkpoint outputs/trocr_small_bi_finetune/best
    python scripts/filter_hwr200.py --keep-percentile 90 --num-beams 1
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from rapidfuzz.distance import Levenshtein
from tqdm import tqdm
from transformers import AutoTokenizer

from src.finetune import Augmenter, build_sources
from src.model import build_processor, build_trocr_small


def _reason(label, pred):
    if len(label) > len(pred) * 1.3:
        return "label_longer (extra text in label?)"
    if len(pred) > len(label) * 1.3:
        return "pred_longer (text missing from label?)"
    return "mismatch"


@torch.no_grad()
def _predict(model, processor, paths, device, num_beams, max_len, batch_size, desc):
    preds = []
    for i in tqdm(range(0, len(paths), batch_size), desc=desc, leave=False):
        imgs = [Image.open(p).convert("RGB") for p in paths[i:i + batch_size]]
        pv = processor(images=imgs, return_tensors="pt").pixel_values.to(device)
        ids = model.generate(pv, num_beams=num_beams, max_length=max_len)
        preds += processor.tokenizer.batch_decode(ids, skip_special_tokens=True)
    return preds


def _score_split(ds, model, processor, device, args, max_len, desc):
    samples = getattr(ds, "samples", None)
    if samples is None:
        raise SystemExit(f"need a path-based source (pairs/tsv), got {type(ds).__name__}")
    paths = [str(p) for p, _ in samples]
    labels = [t for _, t in samples]
    preds = _predict(model, processor, paths, device, args.num_beams, max_len, args.batch_size, desc)
    return [{"path": p, "cer": round(float(Levenshtein.normalized_distance(t, h)), 4),
             "label": t, "pred": h, "len_label": len(t), "len_pred": len(h)}
            for p, t, h in zip(paths, labels, preds)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/finetune.yaml")
    ap.add_argument("--checkpoint", default="outputs/trocr_small_bi_finetune/best")
    ap.add_argument("--source", default="hwr200")
    ap.add_argument("--keep-percentile", type=float, default=80.0,
                    help="keep this %% best (lowest-CER) pairs as good; the rest go to bad")
    ap.add_argument("--num-beams", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", type=Path, default=Path("outputs/hwr200_filtered"))
    args = ap.parse_args()

    cfg = OmegaConf.load(ROOT / args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = str(ROOT / args.checkpoint) if (ROOT / args.checkpoint).exists() else args.checkpoint
    max_len = int(cfg.model.max_target_len)

    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model, _ = build_trocr_small(tokenizer, ckpt, max_length=max_len)
    processor = build_processor(tokenizer, ckpt)
    model.to(device).eval()

    spec = OmegaConf.create({args.source: OmegaConf.to_container(cfg.data.sources[args.source], resolve=True)})
    spec[args.source]["enabled"] = True
    clean = Augmenter(train=False)
    srcs = build_sources(spec, ROOT, clean, clean)
    if not srcs:
        raise SystemExit(f"source '{args.source}' produced no data")
    src = srcs[0]

    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    report = {"source": args.source, "checkpoint": args.checkpoint,
              "keep_percentile": args.keep_percentile, "splits": {}}

    for split, ds in (("train", src.train), ("test", src.test)):
        if ds is None or len(ds) == 0:
            print(f"{split}: empty, skipped"); continue
        recs = _score_split(ds, model, processor, device, args, max_len, split)
        cers = np.array([r["cer"] for r in recs])
        thr = float(np.percentile(cers, args.keep_percentile))
        good = [r for r in recs if r["cer"] <= thr]
        bad = sorted((r for r in recs if r["cer"] > thr), key=lambda r: -r["cer"])
        for r in bad:
            r["reason"] = _reason(r["label"], r["pred"])
        (out / f"{args.source}_{split}_good.json").write_text(
            json.dumps(good, ensure_ascii=False, indent=2), encoding="utf-8")
        (out / f"{args.source}_{split}_bad.json").write_text(
            json.dumps(bad, ensure_ascii=False, indent=2), encoding="utf-8")
        report["splits"][split] = {
            "total": len(recs), "good": len(good), "bad": len(bad), "threshold_cer": round(thr, 4),
            "cer_percentiles": {f"p{q}": round(float(np.percentile(cers, q)), 3) for q in (50, 75, 90, 95, 99)}}
        print(f"{split}: total={len(recs)} good={len(good)} bad={len(bad)} thr_cer={thr:.3f}")

    (out / f"{args.source}_filter_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
