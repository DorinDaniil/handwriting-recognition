#!/usr/bin/env python3
"""Download a School Notebooks dataset (ai-forever, COCO: word polygons + transcripts).

Groups pupil_text words into lines by group_id and crops each line keeping the real page —
the words AND the natural paper/spacing between them — while erasing only foreign ink (other
lines, comments) by its segmentation mask, filled with the estimated paper colour and
feathered for smooth, natural transitions. Writes a manifest (crop_path<TAB>text) for
src.finetune.TsvLineDataset.

    python scripts/download/school_notebooks.py --root data/school_notebooks_ru --inspect
    python scripts/download/school_notebooks.py --root data/school_notebooks_ru --preview
    python scripts/download/school_notebooks.py --root data/school_notebooks_ru
"""
import argparse
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

REPO = "ai-forever/school_notebooks_RU"
IMG_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


def _text(ann):
    attrs = ann.get("attributes")
    if isinstance(attrs, dict):
        for key in ("translation", "text", "label"):
            value = attrs.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    value = ann.get("text") or ann.get("translation")
    return value.strip() if isinstance(value, str) and value.strip() else None


def _bbox(ann):
    if ann.get("bbox"):
        return [float(v) for v in ann["bbox"]]
    seg = ann.get("segmentation")
    if not seg:
        return None
    polys = seg if isinstance(seg[0], (list, tuple)) else [seg]
    xs = [float(v) for poly in polys for v in poly[0::2]]
    ys = [float(v) for poly in polys for v in poly[1::2]]
    if not xs:
        return None
    return [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]


def _polys(ann):
    seg = ann.get("segmentation") or []
    polys = seg if seg and isinstance(seg[0], (list, tuple)) else [seg]
    return [list(zip(p[0::2], p[1::2])) for p in polys if p]


def _draw_mask(size, polys, ox, oy, x0, y0, x1, y1):
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for poly in polys:
        xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
        if max(xs) < x0 or min(xs) > x1 or max(ys) < y0 or min(ys) > y1:   # outside crop
            continue
        draw.polygon([(px - ox, py - oy) for px, py in poly], fill=255)
    return mask


def _crop_clean(page, box, own_polys, foreign_polys, margin=0.10):
    """Crop the line keeping the real page (words + natural inter-word spacing + paper) and
    erasing ONLY the foreign text (other lines/comments) with the paper colour, feathered for
    smooth transitions. Own words are never erased."""
    x, y, w, h = box
    pad = int(h * margin)
    x0, y0 = max(0, int(x - pad)), max(0, int(y - pad))
    x1, y1 = min(page.width, int(x + w + pad)), min(page.height, int(y + h + pad))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    crop = page.crop((x0, y0, x1, y1))
    size = crop.size

    own = np.asarray(_draw_mask(size, own_polys, x0, y0, x0, y0, x1, y1), np.float32)
    foreign = np.asarray(_draw_mask(size, foreign_polys, x0, y0, x0, y0, x1, y1), np.float32)
    foreign = np.clip(foreign - own, 0, 255)                  # never touch our own words
    if foreign.max() == 0:
        return crop                                           # nothing foreign to remove

    alpha_img = Image.fromarray(foreign.astype(np.uint8)).filter(ImageFilter.MaxFilter(5))
    alpha = (np.asarray(alpha_img.filter(ImageFilter.GaussianBlur(3)), np.float32) / 255.0)[:, :, None]

    arr = np.asarray(crop, np.float32)
    plain = arr[(foreign == 0) & (own == 0)].reshape(-1, 3)   # paper: not ink, not foreign
    if len(plain):
        bright = plain[plain.mean(1) >= np.percentile(plain.mean(1), 50)]
        bg = np.median(bright if len(bright) else plain, axis=0)
    else:
        bg = np.array([255.0, 255.0, 255.0])
    out = arr * (1 - alpha) + bg * alpha
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def _split_of(name):
    if "test" in name:
        return "test"
    if "train" in name or "val" in name:        # validation folds into train
        return "train"
    return "all"


def _coco_files(snapshot):
    out = []
    for path in snapshot.rglob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict) and "annotations" in data and "images" in data:
            out.append((path, data))
    return out


def _image_source(snapshot):
    zips = sorted(snapshot.rglob("*.zip"))
    if zips:
        zf = zipfile.ZipFile(zips[0])
        index = {Path(n).name: n for n in zf.namelist() if Path(n).suffix.lower() in IMG_EXT}
        return zf, index
    index = {p.name: p for p in snapshot.rglob("*") if p.suffix.lower() in IMG_EXT}
    return None, index


def _open_image(zf, index, file_name):
    key = index.get(Path(file_name).name)
    if key is None:
        return None
    return Image.open(zf.open(key) if zf else key).convert("RGB")


PUPIL_CAT = 0           # pupil_text carries the line transcript
INK_CATS = (0, 1, 2)    # ink to treat as 'foreign' when it belongs to another line
LINE_CAT = 4            # text_line: line geometry only (no transcript)


def _lines(data):
    words = defaultdict(list)                # (image, group) -> [(x0, polys, text)]  pupil only
    ink = defaultdict(list)                  # image -> {group: polys}  all handwriting (for foreign)
    line_box = {}                            # (image, group) -> text_line bbox
    for ann in data["annotations"]:
        cat = int(ann.get("category_id", -1))
        img, grp = str(ann["image_id"]), str(ann.get("group_id"))
        if cat in INK_CATS:
            polys = _polys(ann)
            if polys:
                ink[img].append((grp, polys))
        if cat == PUPIL_CAT:
            text, polys = _text(ann), _polys(ann)
            if text and polys:
                words[(img, grp)].append((min(px for poly in polys for px, _ in poly), polys, text))
        elif cat == LINE_CAT:
            box = _bbox(ann)
            if box:
                line_box[(img, grp)] = box

    by_image = defaultdict(list)             # image -> [(box, own_polys, foreign_polys, text)]
    for (img, grp), items in words.items():
        items.sort(key=lambda t: t[0])       # words left -> right
        text = " ".join(t for _, _, t in items).strip()
        if not text:
            continue
        own = [poly for _, ps, _ in items for poly in ps]
        if (img, grp) in line_box:
            box = line_box[(img, grp)]
        else:
            xs = [px for poly in own for px, _ in poly]
            ys = [py for poly in own for _, py in poly]
            box = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
        foreign = [poly for g, polys in ink[img] if g != grp for poly in polys]
        by_image[img].append((box, own, foreign, text))
    return by_image


def _build(data, zf, index, out_root, split, limit, start):
    id2file = {str(im["id"]): im["file_name"] for im in data["images"]}
    crops_dir = out_root / "crops" / split
    crops_dir.mkdir(parents=True, exist_ok=True)
    records, k = [], start
    for image_id, lines in _lines(data).items():
        page = _open_image(zf, index, id2file.get(image_id, ""))
        if page is None:
            continue
        for box, own, foreign, text in lines:
            crop = _crop_clean(page, box, own, foreign)
            if crop is None:
                continue
            rel = crops_dir / f"{split}_{k:06d}.png"
            crop.save(rel)
            records.append((rel.relative_to(out_root).as_posix(), text))
            k += 1
            if limit and len(records) >= limit:
                return records
    return records


def download(root="data/school_notebooks_ru", repo=REPO, limit=0, preview=False):
    from huggingface_hub import snapshot_download
    snapshot = Path(snapshot_download(repo, repo_type="dataset"))
    out_root = ROOT / root

    cocos = _coco_files(snapshot)
    print(f"  snapshot: {snapshot}")
    print(f"  COCO files: {[p.name for p, _ in cocos] or 'NONE — repo layout differs'}")
    if not cocos:
        return {"root": str(out_root)}
    zf, index = _image_source(snapshot)
    print(f"  images available: {len(index)}")

    by_split = defaultdict(list)
    for path, data in cocos:
        split = _split_of(path.stem.lower())
        recs = _build(data, zf, index, out_root, split,
                      limit or (10 if preview else 0), start=len(by_split[split]))
        print(f"  {path.name}: images={len(data['images'])} lines={len(recs)} -> {split}")
        by_split[split] += recs

    summary = {"root": str(out_root)}
    for split, records in by_split.items():
        summary[split] = len(records)
        if preview:
            for p, t in records[:10]:
                print(f"  {p}\t{t}")
        else:
            (out_root / f"{split}.tsv").write_text(
                "".join(f"{p}\t{t}\n" for p, t in records), encoding="utf-8")
    return summary


def inspect(repo=REPO):
    from huggingface_hub import snapshot_download
    snapshot = Path(snapshot_download(repo, repo_type="dataset"))
    print("snapshot:", snapshot)
    for p in sorted(snapshot.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(snapshot)}  ({p.stat().st_size / 1e6:.1f} MB)")
    cocos = _coco_files(snapshot)
    if not cocos:
        print("no COCO json")
        return
    from collections import Counter
    path, data = cocos[0]
    ann = data["annotations"][0]
    print(f"\ncoco: {path.name} images={len(data['images'])} annotations={len(data['annotations'])}")
    print("image[0]:", data["images"][0])
    print("annotation[0]:", {k: str(v)[:70] for k, v in ann.items()})
    print("categories:", data.get("categories"))
    print("category_id counts:", dict(Counter(str(a.get("category_id")) for a in data["annotations"])))
    sample = {}
    for a in data["annotations"]:
        c = str(a.get("category_id"))
        if c not in sample and _text(a):
            sample[c] = _text(a)
    print("sample text per category:", sample)
    print("with text:", sum(1 for a in data["annotations"] if _text(a)))
    print("with bbox/seg:", sum(1 for a in data["annotations"] if _bbox(a)))
    by_image = _lines(data)
    print("reconstructed lines (first image):", len(next(iter(by_image.values()), [])))
    zf, index = _image_source(snapshot)
    print("images available:", len(index))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/school_notebooks_ru")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--inspect", action="store_true", help="print repo files + COCO schema, then exit")
    args = ap.parse_args()
    if args.inspect:
        inspect(args.repo)
        return
    print("school_notebooks:", download(args.root, args.repo, args.limit, args.preview))


if __name__ == "__main__":
    main()
