#!/usr/bin/env python3
"""End-to-end handwriting OCR: a page image in, recognized text out.

Chains the two models in this repo: DBNet++ (detection/) finds line boxes, then
TrOCR (recognition/) reads each line crop. Both packages are named ``src``, so we
register them under distinct aliases (detsrc / recsrc) to coexist in one process.

    python infer_page.py --image page.jpg
    python infer_page.py --image page.jpg --det-ckpt ... --rec-ckpt ... --device cpu
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
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent
DET_ROOT = REPO / "detection"
REC_ROOT = REPO / "recognition"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _register_pkg(src_dir: Path, alias: str):
    """Import the package at ``src_dir`` under the top-level name ``alias``."""
    spec = importlib.util.spec_from_file_location(
        alias, src_dir / "__init__.py", submodule_search_locations=[str(src_dir)]
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load package at {src_dir}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


_register_pkg(DET_ROOT / "src", "detsrc")
_register_pkg(REC_ROOT / "src", "recsrc")

from detsrc.model import build_model                               # noqa: E402
from detsrc.postprocess import PostprocessConfig, decode_prob_map  # noqa: E402
from recsrc.model import build_processor, build_trocr_small        # noqa: E402


def imread_rgb(path: Path) -> np.ndarray:
    """Read an image as RGB."""
    return np.array(Image.open(path).convert("RGB"))


def _order_quad(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = pts[:, 0] - pts[:, 1]
    return np.stack([pts[np.argmin(s)], pts[np.argmax(d)],
                     pts[np.argmax(s)], pts[np.argmin(d)]]).astype(np.float32)


def expand_quad(quad: np.ndarray, expand_w: float, expand_h: float) -> np.ndarray:
    """Grow a quad from its centre by a fraction of its own width/height, along the
    box's local axes (so tilted boxes grow straight, not along global X/Y)."""
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
        d = p - c
        out[i] = c + np.dot(d, u) * (1.0 + expand_w) * u + np.dot(d, v) * (1.0 + expand_h) * v
    return out.astype(np.float32)


def estimate_bg(rgb: np.ndarray) -> tuple[int, int, int]:
    """Neutral page colour (median over a pixel sample) — usually the paper."""
    flat = rgb.reshape(-1, 3)
    if len(flat) > 200_000:
        flat = flat[np.random.default_rng(0).choice(len(flat), 200_000, replace=False)]
    return tuple(int(v) for v in np.median(flat, axis=0))


def warp_crop(rgb: np.ndarray, quad: np.ndarray, bg: tuple[int, int, int],
              margin_frac: float = 0.08) -> np.ndarray | None:
    """Rectify a (possibly tilted) quad into an upright line crop.

    Maps the quad's own corners onto an upright rectangle instead of taking an
    axis-aligned bbox, so tilted boxes don't swallow neighbouring lines. Out-of-image
    pixels and the margin around the line are filled with the neutral page colour.
    """
    q = _order_quad(quad)
    w = max(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3]))
    h = max(np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[2] - q[1]))
    w, h = int(round(w)), int(round(h))
    if w < 2 or h < 2:
        return None
    tl, tr, br, bl = q
    data = (tl[0], tl[1], bl[0], bl[1], br[0], br[1], tr[0], tr[1])  # PIL QUAD: UL, LL, LR, UR
    src = Image.fromarray(rgb)
    try:
        line = src.transform((w, h), Image.QUAD, data, resample=Image.BILINEAR, fillcolor=bg)
    except (TypeError, ValueError):
        line = src.transform((w, h), Image.QUAD, data, resample=Image.BILINEAR)
    if margin_frac > 0:
        m = max(1, int(round(h * margin_frac)))
        canvas = Image.new("RGB", (w + 2 * m, h + 2 * m), bg)
        canvas.paste(line, (m, m))
        line = canvas
    return np.asarray(line)


def reading_order(quads: list[np.ndarray]) -> list[int]:
    """Indices of ``quads`` in reading order: rows top->bottom, left->right."""
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


def pick_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA unavailable -> using cpu", file=sys.stderr)
        return torch.device("cpu")
    return torch.device(requested)


class LineDetector:
    """DBNet++ line detector. detect(rgb) -> (boxes, scores) with raw detector quads.

    expand_w / expand_h are not applied here — they grow the box only at crop time
    (recognize_page), so the boxes returned/visualized stay the clean detector output.
    """

    def __init__(self, config: Path, ckpt: Path, device: torch.device, use_ema: bool = True,
                 # expand_w: float = 0.08, expand_h: float = 0.24):
                 expand_w: float = 0.04, expand_h: float = 0.16):
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

    def _preprocess(self, rgb: np.ndarray) -> tuple[torch.Tensor, float]:
        """Letterbox to a square: longest side -> self.size, pad bottom-right with 0."""
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
    def detect(self, rgb: np.ndarray) -> tuple[list[np.ndarray], list[float]]:
        h0, w0 = rgb.shape[:2]
        x, scale = self._preprocess(rgb)
        prob = self.model(x.to(self.device))["prob"].float().squeeze().cpu().numpy()
        return decode_prob_map(prob, self.pp, scale=scale, pad=(0.0, 0.0), original_size=(w0, h0))


class LineRecognizer:
    """Fine-tuned TrOCR-small. recognize([rgb_crop, ...]) -> [text, ...]."""

    def __init__(self, config: Path, ckpt: Path, device: torch.device,
                 num_beams: int = 4, batch_size: int = 16):
        from transformers import AutoTokenizer

        cfg = OmegaConf.load(config)
        self.max_len = int(cfg.model.max_target_len)
        self.num_beams = num_beams
        self.batch_size = batch_size
        self.device = device

        ckpt = str(ckpt)
        tokenizer = AutoTokenizer.from_pretrained(ckpt)
        model, _ = build_trocr_small(tokenizer, ckpt, max_length=self.max_len)
        self.processor = build_processor(tokenizer, ckpt)
        self.model = model.eval().to(device)
        print(f"[rec] loaded {ckpt} (beams={num_beams}, max_len={self.max_len})")

    @torch.no_grad()
    def recognize(self, crops: list[np.ndarray]) -> list[str]:
        texts: list[str] = []
        for i in range(0, len(crops), self.batch_size):
            chunk = [Image.fromarray(c) for c in crops[i:i + self.batch_size]]
            pixel_values = self.processor(images=chunk, return_tensors="pt").pixel_values
            ids = self.model.generate(pixel_values.to(self.device),
                                      num_beams=self.num_beams, max_length=self.max_len)
            texts += self.processor.tokenizer.batch_decode(ids, skip_special_tokens=True)
        return texts


def recognize_page(image: Path, detector: LineDetector, recognizer: LineRecognizer,
                   return_crops: bool = False):
    """Detect lines, then recognize each one.

    Returns (full_text, lines) with lines = [{box, score, text}, ...] in reading order;
    ``box`` is the raw detector quad. With return_crops=True also returns the aligned
    crops fed to the recognizer (each grown by detector.expand_w/expand_h).
    """
    rgb = imread_rgb(image)
    boxes, scores = detector.detect(rgb)
    if not boxes:
        return ("", [], []) if return_crops else ("", [])

    bg = estimate_bg(rgb)
    crops, kept = [], []
    for idx in reading_order(boxes):
        box = expand_quad(boxes[idx], detector.expand_w, detector.expand_h)
        crop = warp_crop(rgb, box, bg)
        if crop is None:
            continue
        crops.append(crop)
        kept.append(idx)

    texts = recognizer.recognize(crops)
    lines = [{"box": boxes[idx].tolist(), "score": float(scores[idx]), "text": t}
             for idx, t in zip(kept, texts)]
    full_text = "\n".join(ln["text"] for ln in lines)
    return (full_text, lines, crops) if return_crops else (full_text, lines)


def visualize(image: "Path | np.ndarray", lines: list[dict],
              out: Path | None = None, save: bool = False,
              box_color=(220, 30, 30), width: int = 2) -> Image.Image:
    """Draw the detector's line boxes on the page (clean detection, no text).

    Returns the annotated image; writes a file only when save=True.
    """
    img = (Image.fromarray(image) if isinstance(image, np.ndarray)
           else Image.open(image)).convert("RGB")
    draw = ImageDraw.Draw(img)
    for line in lines:
        pts = np.asarray(line["box"], dtype=float).reshape(-1, 2)
        quad = [(int(round(x)), int(round(y))) for x, y in pts]
        draw.line(quad + quad[:1], fill=box_color, width=width)

    if save:
        if out is None:
            raise ValueError("save=True requires out=<path>")
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        img.save(out)
    return img


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """A TTF with Cyrillic coverage if available, else the PIL bitmap default."""
    for name in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                 "C:/Windows/Fonts/arial.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def show_lines(crops: list[np.ndarray], lines: list[dict],
               out: Path | None = None, save: bool = False,
               width: int = 820, text_color=(0, 0, 160), pad: int = 8) -> Image.Image:
    """Per-line inspection: each crop fed to the recognizer (grey-bordered) with its
    recognized text above it, stacked into one image.

    ``crops`` and ``lines`` are the aligned outputs of recognize_page(return_crops=True).
    Returns the image; writes a file only when save=True.
    """
    font = _load_font(20)
    lh = sum(font.getmetrics())
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))

    rows = []
    for i, (crop, line) in enumerate(zip(crops, lines)):
        c = (Image.fromarray(crop) if isinstance(crop, np.ndarray) else crop).convert("RGB")
        if c.width > width:
            c = c.resize((width, max(1, round(c.height * width / c.width))))
        rows.append((c, f"{i + 1}: {line.get('text', '')}"))

    if not rows:
        return Image.new("RGB", (width, 40), (255, 255, 255))

    content_w = max(max(c.width for c, _ in rows),
                    max(int(measure.textlength(lbl, font=font)) for _, lbl in rows))
    canvas = Image.new("RGB", (content_w + 2 * pad,
                               sum(lh + pad + c.height + 2 * pad for c, _ in rows) + pad),
                       (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    y = pad
    for c, label in rows:
        draw.text((pad, y), label, font=font, fill=text_color)
        y += lh + pad
        canvas.paste(c, (pad, y))
        draw.rectangle([pad, y, pad + c.width - 1, y + c.height - 1], outline=(190, 190, 190))
        y += c.height + 2 * pad

    if save:
        if out is None:
            raise ValueError("save=True requires out=<path>")
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out)
    return canvas


def _resolve(path: Path | None, default: Path, what: str) -> Path:
    p = Path(path) if path else default
    if not p.exists():
        raise FileNotFoundError(f"{what} not found: {p}\n  pass an explicit path with its flag.")
    return p


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Handwritten page -> recognized text (detection + TrOCR).")
    p.add_argument("--image", type=Path, required=True, help="page image to recognize")
    p.add_argument("--det-config", type=Path, default=DET_ROOT / "config.yaml")
    p.add_argument("--det-ckpt", type=Path, default=DET_ROOT / "outputs/dbnetpp_r18_hwr/best.pt")
    p.add_argument("--rec-config", type=Path, default=REC_ROOT / "configs/finetune.yaml")
    p.add_argument("--rec-ckpt", type=Path, default=REC_ROOT / "outputs/trocr_small_bi_finetune/best")
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-beams", type=int, default=4)
    p.add_argument("--no-ema", dest="use_ema", action="store_false", default=True,
                   help="use raw detection weights instead of EMA")
    p.add_argument("--expand-w", type=float, default=0.03,
                   help="grow each crop by this fraction of box width (anti-clipping)")
    p.add_argument("--expand-h", type=float, default=0.06,
                   help="grow each crop by this fraction of box height (anti-clipping)")
    p.add_argument("--out", type=Path, default=None, help="write recognized text here")
    p.add_argument("--save-json", type=Path, default=None, help="write per-line boxes + text")
    p.add_argument("--save-viz", type=Path, default=None, help="write page with detection boxes drawn")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    image = _resolve(args.image, args.image, "image")
    det_cfg = _resolve(args.det_config, args.det_config, "detection config")
    det_ckpt = _resolve(args.det_ckpt, args.det_ckpt, "detection checkpoint")
    rec_cfg = _resolve(args.rec_config, args.rec_config, "recognition config")
    rec_ckpt = _resolve(args.rec_ckpt, args.rec_ckpt, "recognition checkpoint")

    device = pick_device(args.device)
    detector = LineDetector(det_cfg, det_ckpt, device, use_ema=args.use_ema,
                            expand_w=args.expand_w, expand_h=args.expand_h)
    recognizer = LineRecognizer(rec_cfg, rec_ckpt, device, num_beams=args.num_beams)

    full_text, lines = recognize_page(image, detector, recognizer)
    print(f"\n[done] {len(lines)} lines\n" + "=" * 60)
    print(full_text)
    print("=" * 60)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(full_text, encoding="utf-8")
        print(f"[saved] text  -> {args.out}")
    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(
            json.dumps({"image": str(image), "lines": lines}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[saved] json  -> {args.save_json}")
    if args.save_viz:
        visualize(image, lines, out=args.save_viz, save=True)
        print(f"[saved] viz   -> {args.save_viz}")


if __name__ == "__main__":
    main()
