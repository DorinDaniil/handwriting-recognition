#!/usr/bin/env python3
"""Build a word-level manifest from IMGUR5K (English handwriting, in-the-wild).

Does everything end to end: clones the IMGUR5K repo, downloads the images with THEIR tool,
crops each rotated word box into an upright crop (PIL only), writes <out>/<split>.tsv
(crop_path<TAB>word) for src.finetune.TsvLineDataset, then deletes the clone.

    python scripts/download/imgur5k.py --out data/imgur5k            # full auto
    python scripts/download/imgur5k.py --out data/imgur5k --keep     # keep the clone
    python scripts/download/imgur5k.py --out data/imgur5k --root /existing/clone   # reuse a clone

Requires `git` and the IMGUR5K downloader's deps (`requests`). Crops go to <out> (must be
writable); the temporary clone lives next to <out> and is removed at the end unless --keep.
"""
import argparse
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

REPO_URL = "https://github.com/facebookresearch/IMGUR5K-Handwriting-Dataset"
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def _order_quad(pts):
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s, d = pts.sum(1), pts[:, 0] - pts[:, 1]
    return np.stack([pts[s.argmin()], pts[d.argmax()], pts[s.argmax()], pts[d.argmin()]])


def _corners(xc, yc, w, h, angle_deg):
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    return [(xc + dx * ca - dy * sa, yc + dx * sa + dy * ca)
            for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2))]


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


def _load_ann(repo: Path, split: str) -> dict:
    p = repo / "dataset_info" / f"imgur5k_annotations_{split}.json"
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
            if not word or word == "." or box == ".":
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


def _patch_downloader(repo: Path):
    """Their old download_imgur5k.py needs a few fixes to run on modern stacks: a User-Agent
    header (Imgur 403s without one, PR #17), deprecated numpy aliases like np.str (removed in
    NumPy >= 1.24), and np.loadtxt(delimiter="\\n") (newline delimiter now rejected)."""
    f = repo / "download_imgur5k.py"
    src = orig = f.read_text(encoding="utf-8")
    src = src.replace(
        "requests.get(image_url).content",
        'requests.get(image_url, headers={"User-Agent": "Mozilla/5.0"}).content',
    )
    src = re.sub(r"\bnp\.(str|int|float|bool|object|long|unicode)(?!\w)", r"\1", src)
    src = src.replace(r'delimiter="\n"', "delimiter=None").replace(r"delimiter='\n'", "delimiter=None")
    if src != orig:
        f.write_text(src, encoding="utf-8")
        print("patched download_imgur5k.py (User-Agent + numpy aliases + loadtxt delimiter)")


def clone_and_fetch(work: Path) -> Path:
    """Shallow-clone the repo into `work` and download the images with their tool."""
    repo = work / "IMGUR5K-Handwriting-Dataset"
    if not repo.exists():
        print(f"cloning {REPO_URL} -> {repo}")
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(repo)], check=True)
    images = repo / "images"
    if not (images.exists() and any(images.iterdir())):
        _patch_downloader(repo)                      # User-Agent (Imgur 403) + np.str (NumPy>=1.24)
        print("downloading images (their tool — this fetches ~8k images, slow)...")
        subprocess.run([sys.executable, "download_imgur5k.py",
                        "--dataset_info_dir", "dataset_info", "--output_dir", "images"],
                       cwd=repo, check=True)
    return repo


def download(out="data/imgur5k", root=None, work=None, split="all", keep=False, limit=0, preview=False):
    out_root = Path(out) if Path(out).is_absolute() else ROOT / out
    out_root.mkdir(parents=True, exist_ok=True)
    if preview:
        print("imgur5k: preview skipped (needs the full ~8k-image download)")
        return {"out": str(out_root)}

    clone_area = None
    if root:
        repo = Path(root)
    else:
        clone_area = Path(work) if work else out_root.parent / "_imgur5k_src"
        clone_area.mkdir(parents=True, exist_ok=True)
        repo = clone_and_fetch(clone_area)

    try:
        def crops(sp):
            return build_split(repo / "images", _load_ann(repo, sp), out_root, sp, limit)
        if split == "all":
            written = {"train": crops("train") + crops("val"),   # val folded into train
                       "test": crops("test")}
        else:
            written = {split: crops(split)}
        summary = {"out": str(out_root)}
        for name, recs in written.items():
            (out_root / f"{name}.tsv").write_text(
                "".join(f"{p}\t{t}\n" for p, t in recs), encoding="utf-8")
            summary[name] = len(recs)
            print(f"{name}: {len(recs)} word crops -> {out_root / f'{name}.tsv'}")
        return summary
    finally:
        if clone_area and not keep:
            shutil.rmtree(clone_area, ignore_errors=True)
            print(f"removed clone: {clone_area}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/imgur5k", help="output dir for crops + tsv (writable!)")
    ap.add_argument("--split", default="all", choices=["train", "val", "test", "all"],
                    help="'all' folds val into train.tsv and writes test.tsv separately")
    ap.add_argument("--root", default=None, help="reuse an existing clone instead of cloning")
    ap.add_argument("--work", default=None, help="where to clone (default: <out>/../_imgur5k_src)")
    ap.add_argument("--keep", action="store_true", help="do not delete the clone afterwards")
    ap.add_argument("--limit", type=int, default=0, help="cap crops (debug)")
    args = ap.parse_args()
    print("imgur5k:", download(args.out, args.root, args.work, args.split, args.keep, args.limit))


if __name__ == "__main__":
    main()
