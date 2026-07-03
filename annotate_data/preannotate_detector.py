#!/usr/bin/env python3
"""Pre-annotate line DETECTION on a dataset with MY DBNet++ model, written in the annotator's
json format so I can fix the boxes on top by hand. Self-contained: loads only detection/src
(no recognition / transformers deps).

For every image under --data that has a sibling <stem>.txt (my text markup) and no <stem>.json
yet, it runs the detector, orders the boxes in reading order, and writes <stem>.json:

    {"image": "...", "width": W, "height": H, "source": "dbnetpp",
     "lines": [{"order": 1, "polygon": [[x,y],[x,y],[x,y],[x,y]], "text": "...", "score": ...}]}

Text is pulled from my <stem>.txt line-by-line in reading order. If the detector's line count
!= my text's line count, that's fine — extra boxes get empty text, extra text lines are
dropped, and the page is flagged; fix it in the annotator. My <stem>.txt is NOT modified.
Never touches data/cleaned or *_cropped; existing <stem>.json is skipped (unless --overwrite).

    python preannotate_detector.py --data data --det-ckpt detection/outputs/dbnetpp_r34_hwr/best.pt
    python preannotate_detector.py --data data --device cpu --limit 5 --overwrite
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

REPO = Path(__file__).resolve().parent
DET_ROOT = REPO / "detection"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SKIP_DIRS = {".ipynb_checkpoints"}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _register_pkg(src_dir: Path, alias: str):
    spec = importlib.util.spec_from_file_location(
        alias, src_dir / "__init__.py", submodule_search_locations=[str(src_dir)])
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load package at {src_dir}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


_register_pkg(DET_ROOT / "src", "detsrc")
from detsrc.model import build_model                               # noqa: E402
from detsrc.postprocess import PostprocessConfig, decode_prob_map  # noqa: E402


# --------------------------------------------------------------------------- geometry helpers
def imread_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def _order_quad(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = pts[:, 0] - pts[:, 1]
    return np.stack([pts[np.argmin(s)], pts[np.argmax(d)],
                     pts[np.argmax(s)], pts[np.argmin(d)]]).astype(np.float32)


def expand_quad(quad: np.ndarray, expand_w: float, expand_h: float) -> np.ndarray:
    """Grow a quad from its centre along its own local axes by a fraction of w/h."""
    if expand_w == 0.0 and expand_h == 0.0:
        return np.asarray(quad, dtype=np.float32).reshape(4, 2)
    q = _order_quad(quad).astype(np.float64)
    c = q.mean(axis=0)
    u, v = q[1] - q[0], q[3] - q[0]
    wl, hl = np.linalg.norm(u), np.linalg.norm(v)
    if wl < 1e-6 or hl < 1e-6:
        return q.astype(np.float32)
    u, v = u / wl, v / hl
    out = np.empty_like(q)
    for i, p in enumerate(q):
        dd = p - c
        out[i] = c + np.dot(dd, u) * (1.0 + expand_w) * u + np.dot(dd, v) * (1.0 + expand_h) * v
    return out.astype(np.float32)


def reading_order(quads: list[np.ndarray]) -> list[int]:
    """Indices of quads in reading order: rows top->bottom, left->right."""
    if not quads:
        return []
    boxes = []
    for i, q in enumerate(quads):
        ys = q[:, 1]
        boxes.append((i, float(q[:, 0].min()), float(ys.min()), float(ys.max()),
                      float((ys.min() + ys.max()) / 2)))
    line_h = float(np.median([b[3] - b[2] for b in boxes])) or 1.0
    rows: list[dict] = []
    for b in sorted(boxes, key=lambda b: b[4]):
        for row in rows:
            if abs(b[4] - row["yc"]) < 0.6 * line_h:
                row["items"].append(b)
                row["yc"] = float(np.mean([x[4] for x in row["items"]]))
                break
        else:
            rows.append({"yc": b[4], "items": [b]})
    order: list[int] = []
    for row in sorted(rows, key=lambda r: r["yc"]):
        order.extend(b[0] for b in sorted(row["items"], key=lambda b: b[1]))
    return order


def pick_device(req: str) -> torch.device:
    if req == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA unavailable -> cpu", file=sys.stderr)
        return torch.device("cpu")
    return torch.device(req)


class LineDetector:
    """DBNet++ line detector. detect(rgb) -> (quads, scores) with raw detector quads."""

    def __init__(self, config: Path, ckpt: Path, device: torch.device, use_ema: bool = True,
                 expand_w: float = 0.0, expand_h: float = 0.0):
        cfg = OmegaConf.load(config)
        cfg.model.backbone.pretrained = False
        self.size = int(cfg.data.image_size)
        self.expand_w, self.expand_h = float(expand_w), float(expand_h)
        self.pp = PostprocessConfig(
            thresh=float(cfg.postprocess.thresh),
            box_thresh=float(cfg.postprocess.box_thresh),
            unclip_ratio=float(cfg.postprocess.unclip_ratio),
            max_candidates=int(cfg.postprocess.max_candidates),
            min_size=int(cfg.postprocess.min_size),
        )
        self.device = device
        model = build_model(cfg)
        state = torch.load(ckpt, map_location="cpu")
        # weights = state["ema"] if (use_ema and state.get("ema") is not None) else state["model"]
        weights = state
        model.load_state_dict(weights)
        self.model = model.eval().to(device)
        tag = "EMA" if (use_ema and state.get("ema") is not None) else "raw"
        print(f"[det] loaded {Path(ckpt).name} (epoch {state.get('epoch')}, {tag} weights)")

    def _preprocess(self, rgb: np.ndarray):
        h0, w0 = rgb.shape[:2]
        scale = self.size / max(h0, w0)
        nh, nw = int(round(h0 * scale)), int(round(w0 * scale))
        resized = np.asarray(Image.fromarray(rgb).resize((nw, nh), Image.BILINEAR))
        canvas = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        canvas[:nh, :nw] = resized
        x = (canvas.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        x = np.transpose(x, (2, 0, 1))[None]
        return torch.from_numpy(x), scale

    @torch.no_grad()
    def detect(self, rgb: np.ndarray):
        h0, w0 = rgb.shape[:2]
        x, scale = self._preprocess(rgb)
        prob = self.model(x.to(self.device))["prob"].float().squeeze().cpu().numpy()
        return decode_prob_map(prob, self.pp, scale=scale, pad=(0.0, 0.0), original_size=(w0, h0))


# --------------------------------------------------------------------------- pre-annotation
def txt_lines(img: Path):
    t = img.with_suffix(".txt")
    if not t.exists():
        return None
    return [ln.rstrip("\n") for ln in t.read_text(encoding="utf-8").splitlines() if ln.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", help="dataset root (images + sibling <stem>.txt)")
    ap.add_argument("--det-config", type=Path, default=DET_ROOT / "config.yaml")
    ap.add_argument("--det-ckpt", type=Path, default=DET_ROOT / "outputs/dbnetpp_r34_hwr/best.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-ema", dest="use_ema", action="store_false", default=True)
    ap.add_argument("--expand-w", type=float, default=0.0, help="grow box width fraction (0 = raw)")
    ap.add_argument("--expand-h", type=float, default=0.0, help="grow box height fraction (0 = raw)")
    ap.add_argument("--overwrite", action="store_true", help="redo even if <stem>.json exists")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.data)
    device = pick_device(args.device)
    det = LineDetector(args.det_config, args.det_ckpt, device, use_ema=args.use_ema,
                       expand_w=args.expand_w, expand_h=args.expand_h)

    imgs = [p for p in sorted(root.rglob("*"))
            if p.suffix.lower() in IMG_EXT and not any(d in p.parts for d in SKIP_DIRS)
            and p.with_suffix(".txt").exists()]
    done = skipped = mism = 0
    for img in imgs:
        jp = img.with_suffix(".json")
        if jp.exists() and not args.overwrite:
            skipped += 1
            continue
        if args.limit and done >= args.limit:
            break

        rgb = imread_rgb(img)
        H, W = rgb.shape[:2]
        boxes, scores = det.detect(rgb)
        order = reading_order(boxes)
        my = txt_lines(img) or []

        lines = []
        for i, idx in enumerate(order):
            quad = _order_quad(expand_quad(boxes[idx], det.expand_w, det.expand_h))
            lines.append({"order": i + 1,
                          "polygon": [[int(round(x)), int(round(y))] for x, y in quad],
                          "text": my[i] if i < len(my) else "",
                          "score": round(float(scores[idx]), 4)})

        jp.write_text(json.dumps({"image": img.name, "width": W, "height": H,
                                  "source": "dbnetpp", "lines": lines},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
        done += 1
        if len(order) == len(my):
            tag = "1:1 текст подставлен"
        else:
            tag = f"строк детектора {len(order)} vs текста {len(my)} — поправь"
            mism += 1
        print(f"  ok  {img.relative_to(root)}  [{tag}]")

    print(f"\ndone={done}  skipped(existing json)={skipped}  mismatch={mism}")


if __name__ == "__main__":
    main()
