#!/usr/bin/env python3
"""End-to-end page evaluation: the unchanged line detector + this recogniser.

Same protocol as kaliningrad_essays_eval.py (target = <image>.txt, lines joined with spaces,
jiwer CER / WER per page, mean over pages), and the result goes into the same json so the runs
sit side by side. cer / wer use the raw target; cer_norm / wer_norm compare after the
recogniser's text normalisation (quotes, dashes, spaces).

Pages the recogniser was trained on (the train part of the `pages` sources in --config) are
left out unless --all is given.

    python scripts/eval_pages.py --data /data/kaliningrad_essays --rec-ckpt outputs/finetune/best.pt
    python scripts/eval_pages.py --data ... --rec-ckpt ... --head auto --device cpu
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path[:0] = [str(ROOT), str(REPO)]

import jiwer
import numpy as np
from omegaconf import OmegaConf
from PIL import Image

from htr.geometry import Frame, crop_line, paper_colour
from htr.predictor import Recognizer
from htr.sources import read_labels, split_pages
from htr.text import normalize
from infer_page import LineDetector, pick_device, reading_order

IMAGES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def page_text(text: str) -> str:
    return " ".join(line.strip() for line in text.splitlines() if line.strip())


def trained_pages(config) -> set:
    cfg, pages = OmegaConf.load(config), set()
    for spec in cfg.data.sources.values():
        if spec.kind != "pages" or not spec.get("enabled", True):
            continue
        root = ROOT / spec.root
        if not (root / spec.get("labels", "labels.txt")).exists():
            continue
        labelled = read_labels(root / spec.get("labels", "labels.txt"), root)
        held_out = split_pages(labelled, spec.get("seed", 42), spec.get("test_frac", 0.1))
        pages |= {str(Path(p).resolve()) for p in labelled if p not in held_out}
    return pages


def read_page(image: Path, detector, reader) -> str:
    page = Image.open(image).convert("RGB")
    boxes, _ = detector.detect(np.array(page))
    fill = paper_colour(page)
    crops = [crop_line(page, boxes[i], reader.frame, fill) for i in reading_order(boxes)]
    return "\n".join(reader.recognize([c for c in crops if c is not None]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="folder with images and <image>.txt targets")
    ap.add_argument("--rec-ckpt", type=Path, required=True)
    ap.add_argument("--head", default="decoder", choices=["ctc", "decoder", "auto"])
    ap.add_argument("--det-config", type=Path, default=REPO / "detection/config.yaml")
    ap.add_argument("--det-ckpt", type=Path, default=REPO / "detection/outputs/dbnetpp_r34_hwr/best.pt")
    ap.add_argument("--config", default=str(ROOT / "configs/finetune.yaml"))
    ap.add_argument("--all", action="store_true", help="also evaluate on pages used for training")
    ap.add_argument("--frame", type=float, nargs=2, metavar=("X", "Y"),
                    help="crop margins around a detector box, in box heights (default: saved with the checkpoint)")
    ap.add_argument("--out", type=Path, default=REPO / "eval_metrics_kaliningrad_essays.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    pages = [p for p in sorted(args.data.rglob("*")) if p.suffix.lower() in IMAGES
             and ".ipynb_checkpoints" not in p.parts and p.with_suffix(".txt").exists()]
    if not args.all:
        seen = trained_pages(args.config)
        kept = [p for p in pages if str(p.resolve()) not in seen]
        print(f"{len(pages) - len(kept)} training pages left out")
        pages = kept
    if not pages:
        raise SystemExit(f"no pages to evaluate under {args.data.resolve()}")

    device = pick_device(args.device)
    detector = LineDetector(args.det_config, args.det_ckpt, device)
    reader = Recognizer(args.rec_ckpt, device, args.head)
    if args.frame:
        reader.frame = Frame.uniform(*args.frame)

    rows, per_page = [], {}
    for i, image in enumerate(pages, 1):
        ref = page_text(image.with_suffix(".txt").read_text(encoding="utf-8"))
        if not ref:
            continue
        hyp = page_text(read_page(image, detector, reader))
        raw = (jiwer.cer(ref, hyp), jiwer.wer(ref, hyp))
        same = (jiwer.cer(normalize(ref), normalize(hyp)), jiwer.wer(normalize(ref), normalize(hyp)))
        rows.append(raw + same)
        per_page[image.relative_to(args.data).as_posix()] = {"cer": round(raw[0], 4), "wer": round(raw[1], 4)}
        print(f"[{i}/{len(pages)}] {image.name}  CER={raw[0]:.4f} WER={raw[1]:.4f}")

    mean, std = np.mean(rows, axis=0), np.std(rows, axis=0)
    print(f"\n{args.data.name} | pages={len(rows)}  CER={mean[0]:.4f}±{std[0]:.4f}  WER={mean[1]:.4f}±{std[1]:.4f}"
          f"  | normalised: CER={mean[2]:.4f}  WER={mean[3]:.4f}")

    store = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {"runs": {}}
    key = (f"{args.data.name} | rec=v2:{'/'.join(args.rec_ckpt.parts[-2:])} | det={args.det_ckpt.stem} "
           f"| head={args.head}" + (f" | frame={args.frame[0]:g},{args.frame[1]:g}" if args.frame else "")
           + ("" if args.all else " | held-out pages"))
    store["runs"][key] = {"samples": len(rows), "cer": round(float(mean[0]), 4), "wer": round(float(mean[1]), 4),
                          "cer_std": round(float(std[0]), 4), "wer_std": round(float(std[1]), 4),
                          "cer_norm": round(float(mean[2]), 4), "wer_norm": round(float(mean[3]), 4),
                          "per_page": per_page}
    args.out.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
