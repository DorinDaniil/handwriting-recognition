#!/usr/bin/env python3
"""End-to-end eval of the pipeline (DBNet++ + TrOCR) on essays.

Target = per-page <stem>.txt, prediction = recognize_page() text. Lines are joined with spaces
and plain CER/WER (jiwer) are computed on the two full-page texts, averaged over pages. Each
model's result is appended to eval_metrics_kaliningrad_essays.json (re-run updates its entry).

    python kaliningrad_essays_eval.py --data data_eval
    python kaliningrad_essays_eval.py --data data_eval --rec-ckpt recognition/outputs/other/best
"""
import argparse
import json
from pathlib import Path

import numpy as np
import jiwer

from infer_page import LineDetector, LineRecognizer, recognize_page, pick_device

REPO = Path(__file__).resolve().parent
DET_CONFIG = REPO / "detection/config.yaml"
REC_CONFIG = REPO / "recognition/configs/finetune.yaml"
OUT = REPO / "eval_metrics_kaliningrad_essays.json"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def page_text(text: str) -> str:
    return " ".join(ln.strip() for ln in text.splitlines() if ln.strip())


def load_store():
    if OUT.exists():
        try:
            d = json.loads(OUT.read_text(encoding="utf-8"))
            if isinstance(d.get("runs"), dict):
                return d
        except Exception:
            pass
    return {"runs": {}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with images + sibling <stem>.txt")
    ap.add_argument("--det-ckpt", type=Path, default=REPO / "detection/outputs/dbnetpp_r34_hwr/best.pt")
    ap.add_argument("--rec-ckpt", type=Path,
                    default=REPO / "recognition/outputs/trocr_small_bi_finetune_with_hwr200_cleaned_v4/best")
    ap.add_argument("--num-beams", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    data = Path(args.data)
    pages = [p for p in sorted(data.rglob("*"))
             if p.suffix.lower() in IMG_EXT and ".ipynb_checkpoints" not in p.parts
             and p.with_suffix(".txt").exists()]
    if not pages:
        raise SystemExit(f"no (image + .txt) pairs under {data.resolve()}")
    print(f"{len(pages)} pages in {data.resolve()}")

    device = pick_device(args.device)
    det = LineDetector(DET_CONFIG, args.det_ckpt, device)
    rec = LineRecognizer(REC_CONFIG, args.rec_ckpt, device, num_beams=args.num_beams)

    cers, wers, per_page = [], [], {}
    for i, img in enumerate(pages, 1):
        pred_text, _ = recognize_page(img, det, rec)
        ref = page_text(img.with_suffix(".txt").read_text(encoding="utf-8"))
        hyp = page_text(pred_text)
        if not ref:
            continue
        cer, wer = float(jiwer.cer(ref, hyp)), float(jiwer.wer(ref, hyp))
        cers.append(cer); wers.append(wer)
        per_page[img.relative_to(data).as_posix()] = {"cer": round(cer, 4), "wer": round(wer, 4)}
        print(f"[{i}/{len(pages)}] {img.name}  CER={cer:.4f} WER={wer:.4f}")

    cer_mean, wer_mean = float(np.mean(cers)), float(np.mean(wers))
    cer_std, wer_std = float(np.std(cers)), float(np.std(wers))
    print(f"\n{data.name} | pages={len(cers)}   "
          f"CER={cer_mean:.4f}±{cer_std:.4f}   WER={wer_mean:.4f}±{wer_std:.4f}")

    key = (f"{data.name} | rec={'/'.join(args.rec_ckpt.parts[-2:])} | "
           f"det={args.det_ckpt.stem} | beams={args.num_beams}")
    store = load_store()
    store["runs"][key] = {"samples": len(cers),
                          "cer": round(cer_mean, 4), "wer": round(wer_mean, 4),
                          "cer_std": round(cer_std, 4), "wer_std": round(wer_std, 4),
                          "per_page": per_page}
    OUT.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved -> {OUT}  (runs: {len(store['runs'])})")

    print(f"\n{'run':<52}{'CER':>8}{'±std':>8}{'WER':>8}{'±std':>8}")
    for rk, rv in store["runs"].items():
        print(f"{rk[:50]:<52}{rv.get('cer', float('nan')):>8.4f}{rv.get('cer_std', float('nan')):>8.4f}"
              f"{rv.get('wer', float('nan')):>8.4f}{rv.get('wer_std', float('nan')):>8.4f}")


if __name__ == "__main__":
    main()
