"""Evaluate a trained DBNet++ checkpoint on a test split.

Usage:
    python evaluate.py \\
        --checkpoint outputs/dbnetpp_r18_hwr/best.pt \\
        --dataset-root /mnt/data/hw_dataset \\
        --labels labels/test_labels.txt \\
        --split splits/test.txt

Optional:
    --config config.yaml         architecture/postprocess defaults (model must match the ckpt)
    --device cuda                or cpu
    --batch-size 8
    --num-workers 4
    --no-ema                     use raw model weights instead of EMA
    --iou-thresh 0.5             single IoU threshold for H-mean
    --iou-range 0.5,0.95,0.05    or sweep over IoUs and report mean (COCO-style)
    --min-score 0.0              ignore GT boxes with teacher score below this
    --pp-thresh / --pp-box-thresh / --pp-unclip / --pp-min-size   override postprocess
    --save-json out.json         dump per-image predictions
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.augmentation import AugConfig
from src.dataset import PaddleOCRDetDataset, detection_collate
from src.model import build_model
from src.postprocess import PostprocessConfig, decode_prob_map
from src.target_gen import TargetConfig
from src.utils import hmean_metric

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("evaluate")


# --------------------------------------------------------------------- args

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # required paths
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True,
                   help="root folder for image files; rel_paths in labels are joined to this")
    p.add_argument("--labels", type=Path, required=True,
                   help="labels.txt in PaddleOCR format: '<rel_path>\\t<JSON list>'")
    p.add_argument("--split", type=Path, required=True,
                   help="text file with one rel_path per line — limits which entries to load")

    # arch / postprocess defaults
    p.add_argument("--config", type=Path, default=Path("config.yaml"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--use-ema", dest="use_ema", action="store_true", default=True)
    p.add_argument("--no-ema", dest="use_ema", action="store_false")

    # eval knobs
    p.add_argument("--iou-thresh", type=float, default=0.5)
    p.add_argument("--iou-range", type=str, default=None,
                   help="comma-separated 'start,stop,step' (COCO-style sweep, e.g. 0.5,0.95,0.05)")
    p.add_argument("--min-score", type=float, default=None,
                   help="override cfg.data.min_score for GT filtering (defaults to config value)")

    # postprocess overrides
    p.add_argument("--pp-thresh", type=float, default=None)
    p.add_argument("--pp-box-thresh", type=float, default=None)
    p.add_argument("--pp-unclip", type=float, default=None)
    p.add_argument("--pp-min-size", type=int, default=None)

    p.add_argument("--save-json", type=Path, default=None,
                   help="if set, dump per-image predictions (boxes + scores) here")

    return p.parse_args()


# ------------------------------------------------------------------- helpers

def _build_pp_cfg(cfg, args) -> PostprocessConfig:
    return PostprocessConfig(
        thresh         = float(args.pp_thresh    if args.pp_thresh    is not None else cfg.postprocess.thresh),
        box_thresh     = float(args.pp_box_thresh if args.pp_box_thresh is not None else cfg.postprocess.box_thresh),
        unclip_ratio   = float(args.pp_unclip    if args.pp_unclip    is not None else cfg.postprocess.unclip_ratio),
        max_candidates = int(cfg.postprocess.max_candidates),
        min_size       = int(args.pp_min_size   if args.pp_min_size   is not None else cfg.postprocess.min_size),
    )


def _iou_grid(args) -> list[float]:
    if args.iou_range is None:
        return [float(args.iou_thresh)]
    a, b, step = (float(x) for x in args.iou_range.split(","))
    out = list(np.round(np.arange(a, b + 1e-9, step), 4))
    return out


# ---------------------------------------------------------------------- main

def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if device.type != args.device:
        logger.warning(f"requested device={args.device} but CUDA unavailable -> using {device}")

    # -- model
    logger.info(f"building model: backbone={cfg.model.backbone.name}")
    model = build_model(cfg)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt["ema"] if (args.use_ema and ckpt.get("ema") is not None) else ckpt["model"]
    model.load_state_dict(state)
    model.eval().to(device)
    logger.info(f"loaded checkpoint: {args.checkpoint.name}  epoch={ckpt.get('epoch')}  "
                f"({'EMA' if args.use_ema and ckpt.get('ema') is not None else 'raw'} weights)")

    # -- dataset (no augmentation, deterministic)
    min_score = float(args.min_score) if args.min_score is not None else float(cfg.data.min_score)
    ds = PaddleOCRDetDataset(
        dataset_root = args.dataset_root,
        labels_txt   = args.labels,
        aug_cfg      = AugConfig(tier="none", image_size=cfg.data.image_size),
        target_cfg   = TargetConfig(
            shrink_ratio = cfg.target.shrink_ratio,
            thresh_min   = cfg.target.thresh_min,
            thresh_max   = cfg.target.thresh_max,
        ),
        split_file   = args.split,
        min_score    = min_score,
        train        = False,
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=detection_collate,
    )
    logger.info(f"test set: {len(ds)} images")

    pp_cfg = _build_pp_cfg(cfg, args)
    logger.info(f"postprocess: thresh={pp_cfg.thresh}  box_thresh={pp_cfg.box_thresh}  "
                f"unclip={pp_cfg.unclip_ratio}  min_size={pp_cfg.min_size}")

    # -- run inference + decode
    pred_polys: list[list[np.ndarray]] = []
    pred_scores: list[list[float]] = []
    gt_polys: list[list[np.ndarray]] = []
    rel_paths: list[str] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="evaluate"):
            images = batch["image"].to(device, non_blocking=True)
            out = model(images)
            prob = out["prob"].float().squeeze(1).cpu().numpy()    # (B, H, W)
            for i in range(images.shape[0]):
                # GT and preds both live in the network's coord frame
                preds, scs = decode_prob_map(prob[i], pp_cfg)
                pred_polys.append(preds)
                pred_scores.append(scs)
                gt_polys.append([np.asarray(g, dtype=np.float32) for g in batch["polys"][i]])
                rel_paths.append(batch["rel_path"][i])

    # -- metrics
    ious = _iou_grid(args)
    print()
    print("=" * 68)
    print(f"{'IoU':>6}  {'Precision':>10}  {'Recall':>8}  {'H-mean':>8}  "
          f"{'TP':>6}  {'FP':>6}  {'FN':>6}")
    print("-" * 68)
    rows = []
    for thr in ious:
        m = hmean_metric(pred_polys, gt_polys, iou_thresh=float(thr))
        rows.append((thr, m))
        print(f"{thr:>6.2f}  {m['precision']:>10.4f}  {m['recall']:>8.4f}  "
              f"{m['hmean']:>8.4f}  {m['tp']:>6d}  {m['fp']:>6d}  {m['fn']:>6d}")
    if len(rows) > 1:
        mean_h = float(np.mean([r[1]["hmean"] for r in rows]))
        print("-" * 68)
        print(f"{'mean':>6}  {'':>10}  {'':>8}  {mean_h:>8.4f}")
    print("=" * 68)
    print(f"images: {len(ds)}  |  predicted boxes: {sum(len(p) for p in pred_polys)}  "
          f"|  GT boxes: {sum(len(g) for g in gt_polys)}")

    summary = {
        "checkpoint": str(args.checkpoint),
        "split":      str(args.split),
        "use_ema":    bool(args.use_ema and ckpt.get("ema") is not None),
        "num_images": len(ds),
        "num_pred_boxes": int(sum(len(p) for p in pred_polys)),
        "num_gt_boxes":   int(sum(len(g) for g in gt_polys)),
        "postprocess": {
            "thresh": pp_cfg.thresh, "box_thresh": pp_cfg.box_thresh,
            "unclip_ratio": pp_cfg.unclip_ratio, "min_size": pp_cfg.min_size,
        },
        "metrics": [{"iou": float(t), **{k: float(v) if not isinstance(v, int) else v
                                          for k, v in m.items()}}
                    for t, m in rows],
    }

    # -- always: short metrics summary next to the checkpoint
    default_summary = args.checkpoint.parent / f"eval_{args.split.stem}.json"
    default_summary.parent.mkdir(parents=True, exist_ok=True)
    with open(default_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"saved metrics: {default_summary}")

    # -- optional: full dump with per-image predictions
    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        records = []
        for rel, preds, scs, gts in zip(rel_paths, pred_polys, pred_scores, gt_polys):
            records.append({
                "rel_path": rel,
                "pred_boxes":  [p.tolist() for p in preds],
                "pred_scores": [float(s) for s in scs],
                "gt_boxes":    [g.tolist() for g in gts],
            })
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump({**summary, "predictions": records}, f, ensure_ascii=False, indent=2)
        logger.info(f"saved full predictions: {args.save_json}")


if __name__ == "__main__":
    main()
