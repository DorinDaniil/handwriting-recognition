#!/usr/bin/env python3
"""Build a word-level manifest from the IMGUR5K handwriting dataset (English, in-the-wild).

IMGUR5K ships as image URLs + rotated word boxes. First clone the repo and download images
with THEIR tool (images aren't redistributed):

    git clone https://github.com/facebookresearch/IMGUR5K-Handwriting-Dataset
    cd IMGUR5K-Handwriting-Dataset
    python download_imgur5k.py --dataset_info_dir dataset_info --output_dir images

Then point this script at the clone — it crops each word box (rotated rect -> upright crop,
PIL only) and writes <out>/<split>.tsv (crop_path<TAB>word) for src.finetune.TsvLineDataset:

    python scripts/download/imgur5k.py --root /path/IMGUR5K-Handwriting-Dataset \
        --split train --out /workspace/.../data/imgur5k
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def _order_quad(pts):
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s, d = pts.sum(1), pts[:, 0] - pts[:, 1]
    return np.stack([pts[s.argmin()], pts[d.argmax()], pts[s.argmax()], pts[d.argmin()]])


def _corners(xc, yc, w, h, angle_deg):
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    out = []
    for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)):
        out.append((xc + dx * ca - dy * sa, yc + dx * sa + dy * ca))
    return out


def _warp(page, xc, yc, w, h, angle):
    w, h = int(round(w)), int(round(h))
    if w < 4 or h < 4:
        return None
    tl, tr, br, bl = _order_quad(_corners(xc, yc, w, h, angle))
    data = (tl[0], tl[1], bl[0], bl[1], br[0], br[1], tr[0], tr[1])  # PIL QUAD: UL, LL, LR, UR
    try:
        return page.transform((w, h), Image.QUAD, data, resample=Image.BILINEAR)
    except (TypeError, ValueError):
        return None


def _load_ann(root: Path, split: str) -> dict:
    p = root / "dataset_info" / f"imgur5k_annotations_{split}.json"
    if not p.exists():
        raise FileNotFoundError(f"annotation file not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def build_split(images: Path, ann: dict, out_root: Path, split: str, limit: int = 0):
    a2m, anns = ann["index_to_ann_map"], ann["ann_id"]
    crops_dir = out_root / "crops" / split
    crops_dir.mkdir(parents=True, exist_ok=True)
    index = {p.stem: p for p in images.iterdir() if p.suffix.lower() in IMG_EXT}

    records, k = [], 0
    for img_id, ann_ids in a2m.items():
        src = index.get(img_id)
        if src is None:
            continue
        try:
            page = Image.open(src).convert("RGB")
        except Exception:
            continue
        for aid in ann_ids:
            word = (anns.get(aid, {}).get("word") or "").strip()
            box = anns.get(aid, {}).get("bounding_box", ".")
            if not word or word == "." or "." == box:
                continue
            try:
                xc, yc, w, h, angle = (float(v) for v in box.strip("[] ").split(","))
            except Exception:
                continue
            crop = _warp(page, xc, yc, w, h, angle)
            if crop is None:
                continue
            rel = crops_dir / f"{img_id}_{k:06d}.png"
            crop.save(rel)
            records.append((str(rel), word.replace("\t", " ").replace("\n", " ")))
            k += 1
            if limit and len(records) >= limit:
                return records
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="cloned IMGUR5K repo (has dataset_info/ + images)")
    ap.add_argument("--images", default=None, help="images dir (default <root>/images)")
    ap.add_argument("--split", default="train", choices=["train", "val", "test", "all"])
    ap.add_argument("--out", default="data/imgur5k", help="output dir for crops + tsv (writable!)")
    ap.add_argument("--limit", type=int, default=0, help="cap crops (debug)")
    args = ap.parse_args()

    root = Path(args.root)
    images = Path(args.images) if args.images else root / "images"
    out_root = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    out_root.mkdir(parents=True, exist_ok=True)

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    for split in splits:
        recs = build_split(images, _load_ann(root, split), out_root, split, args.limit)
        (out_root / f"{split}.tsv").write_text(
            "".join(f"{p}\t{t}\n" for p, t in recs), encoding="utf-8")
        print(f"{split}: {len(recs)} word crops -> {out_root / f'{split}.tsv'}")


if __name__ == "__main__":
    main()
