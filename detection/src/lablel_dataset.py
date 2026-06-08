"""
Pseudo-labeling HWR200 (or any image dataset) with PP-OCRv5_server_det.

Saves bboxes + scores in two formats:
  1) PaddleOCR label.txt      (one line per image, tab-separated, JSON inline)
     — directly usable for PaddleOCR fine-tuning.
  2) COCO detection JSON      — standard format, eats by most detectors.

All image paths are stored RELATIVE to --dataset-root.

Usage:
    python label_hwr200.py --dataset-root /path/to/HWR200 --out-dir ./labels
    python label_hwr200.py --dataset-root /path/to/HWR200 --out-dir ./labels \
        --limit-side-len 1440 --unclip-ratio 2.0 --min-score 0.5

Resumes automatically: already-labeled images are skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from tqdm import tqdm

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ------------------------- file discovery -------------------------

def iter_images(root: Path) -> Iterable[Path]:
    """Recursively yield all image files under root."""
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


# ------------------------- detection -------------------------

def build_detector(args):
    from paddleocr import TextDetection

    # return TextDetection(
    #     model_name=args.model_name,
    #     device=args.device,
    #     limit_side_len=args.limit_side_len,
    #     limit_type="max",
    #     thresh=args.thresh,
    #     box_thresh=args.box_thresh,
    #     unclip_ratio=args.unclip_ratio,
    # )
    return TextDetection(
        model_name=args.model_name,
        device=args.device,
    )


def detect_one(det, image_path: Path) -> list[dict]:
    """
    Run detector on one image, return list of:
      {"points": [[x1,y1],...,[x4,y4]], "score": float, "transcription": ""}
    Coordinates are in ORIGINAL image pixels (PaddleOCR already rescales them back).
    """
    result = det.predict(str(image_path))[0]
    polys = result.get("dt_polys", [])
    scores = result.get("dt_scores", [None] * len(polys))

    out = []
    for poly, score in zip(polys, scores):
        poly = np.asarray(poly).reshape(-1, 2)
        out.append(
            {
                "transcription": "",  # empty — detector only
                "points": [[int(round(x)), int(round(y))] for x, y in poly],
                "score": float(score) if score is not None else None,
            }
        )
    return out


# ------------------------- label IO -------------------------

def read_existing(label_txt: Path) -> set[str]:
    """Return set of relative paths already present in label.txt (for resume)."""
    if not label_txt.exists():
        return set()
    done = set()
    with open(label_txt, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            rel = line.split("\t", 1)[0]
            done.add(rel)
    return done


def append_label_line(label_txt: Path, rel_path: str, boxes: list[dict]):
    with open(label_txt, "a", encoding="utf-8") as f:
        f.write(rel_path + "\t" + json.dumps(boxes, ensure_ascii=False) + "\n")


# ------------------------- COCO conversion -------------------------

def label_txt_to_coco(
    label_txt: Path,
    dataset_root: Path,
    coco_path: Path,
    min_score: float = 0.0,
):
    """Convert PaddleOCR label.txt to COCO detection JSON."""
    images, annotations = [], []
    ann_id = 1
    cat = [{"id": 1, "name": "text_line", "supercategory": "text"}]

    with open(label_txt, "r", encoding="utf-8") as f:
        for img_id, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            rel, payload = line.split("\t", 1)
            boxes = json.loads(payload)

            abs_path = dataset_root / rel
            try:
                with Image.open(abs_path) as im:
                    w, h = im.size
            except Exception:
                w, h = 0, 0

            images.append(
                {
                    "id": img_id,
                    "file_name": rel,
                    "width": w,
                    "height": h,
                }
            )

            for b in boxes:
                score = b.get("score") or 0.0
                if score < min_score:
                    continue
                pts = np.array(b["points"], dtype=float)
                x, y = pts[:, 0].min(), pts[:, 1].min()
                bw = pts[:, 0].max() - x
                bh = pts[:, 1].max() - y
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": 1,
                        "bbox": [float(x), float(y), float(bw), float(bh)],
                        "area": float(bw * bh),
                        "iscrowd": 0,
                        "segmentation": [pts.flatten().tolist()],  # polygon
                        "score": float(score),
                    }
                )
                ann_id += 1

    coco = {"images": images, "annotations": annotations, "categories": cat}
    with open(coco_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False)


# ------------------------- main -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True,
                        help="Root directory of the dataset (HWR200).")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Where to save labels.")
    parser.add_argument("--model-name", default="PP-OCRv5_server_det")
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument("--limit-side-len", type=int, default=1440,
                        help="Bigger = more accurate on small text, slower, more VRAM.")
    parser.add_argument("--thresh", type=float, default=0.3)
    parser.add_argument("--box-thresh", type=float, default=0.5)
    parser.add_argument("--unclip-ratio", type=float, default=1.8,
                        help="Expand factor; 1.8-2.2 covers text tails well on Russian handwriting.")
    parser.add_argument("--min-score", type=float, default=0.0,
                        help="Filter boxes with score below this (applied at COCO export only).")
    parser.add_argument("--coco", action="store_true",
                        help="Also write COCO JSON.")
    args = parser.parse_args()

    dataset_root: Path = args.dataset_root.resolve()
    out_dir: Path = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    label_txt = out_dir / "labels.txt"
    failed_log = out_dir / "failed.txt"

    # resume support
    done = read_existing(label_txt)
    print(f"Already labeled: {len(done)} images (will skip).")

    all_images = list(iter_images(dataset_root))
    print(f"Total images under {dataset_root}: {len(all_images)}")

    det = build_detector(args)

    pbar = tqdm(all_images, desc="labeling")
    n_ok, n_skip, n_fail = 0, 0, 0
    for img_path in pbar:
        rel = os.path.relpath(img_path, dataset_root)
        if rel in done:
            n_skip += 1
            continue
        try:
            boxes = detect_one(det, img_path)
            append_label_line(label_txt, rel, boxes)
            n_ok += 1
            pbar.set_postfix(ok=n_ok, skip=n_skip, fail=n_fail, boxes=len(boxes))
        except Exception as e:
            n_fail += 1
            with open(failed_log, "a", encoding="utf-8") as f:
                f.write(f"{rel}\t{type(e).__name__}: {e}\n")
            traceback.print_exc()

    print(f"\nDone: ok={n_ok}, skipped={n_skip}, failed={n_fail}")
    print(f"Label file: {label_txt}")
    if failed_log.exists():
        print(f"Failures:   {failed_log}")

    if args.coco:
        coco_path = out_dir / "labels_coco.json"
        print(f"Exporting COCO → {coco_path}")
        label_txt_to_coco(label_txt, dataset_root, coco_path, min_score=args.min_score)
        print(f"COCO file:  {coco_path}")


if __name__ == "__main__":
    main()