#!/usr/bin/env python3
"""End-to-end evaluation of the full pipeline (DBNet++ detection + TrOCR recognition) on the
Kaliningrad essays dataset. Target = the per-page <stem>.txt (our markup). Prediction =
recognize_page() full-page text. Metrics are averaged PER PAGE.

Same metric names & libs as recognition/scripts/eval_finetune.py — cer, wer (jiwer),
nes_char, nes_word (rapidfuzz normalized similarity). Results ACCUMULATE across models: the
output JSON keeps one entry per run (keyed by model/config), so a new checkpoint is APPENDED
rather than overwriting previous ones; re-running the same config updates its own entry. Each
run stores page-averaged mean+std AND the full per-page results.

    python kaliningrad_essays_eval.py --data data_eval
    python kaliningrad_essays_eval.py --data data_eval --rec-ckpt .../other/best   # appended
    python kaliningrad_essays_eval.py --data data_eval --join-hyphen --csv pages.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import jiwer
from rapidfuzz.distance import Levenshtein

from infer_page import LineDetector, LineRecognizer, recognize_page, pick_device

REPO = Path(__file__).resolve().parent
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SKIP_DIRS = {".ipynb_checkpoints"}
_WS = re.compile(r"\s+")
METRICS = ("cer", "wer", "nes_char", "nes_word")   # same names as eval_finetune.py


def join_lines(lines, dehyphenate: bool) -> str:
    text = ""
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        if dehyphenate and text.endswith("-"):
            text = text[:-1] + ln
        else:
            text = (text + " " + ln) if text else ln
    return text


def flat(s: str) -> str:
    return _WS.sub(" ", s.replace("\n", " ")).strip()


def page_metrics(ref: str, hyp: str) -> dict:
    """Per-page cer/wer/nes_char/nes_word — identical formulas to src/finetune/metrics.py."""
    return {"cer": float(jiwer.cer(ref, hyp)),
            "wer": float(jiwer.wer(ref, hyp)),
            "nes_char": float(Levenshtein.normalized_similarity(ref, hyp)),
            "nes_word": float(Levenshtein.normalized_similarity(ref.split(), hyp.split()))}


def load_store(path: Path) -> dict:
    """Load the accumulating results file. Old flat-format files are preserved under runs['legacy']."""
    if not path.exists():
        return {"runs": {}}
    try:
        prev = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"runs": {}}
    if isinstance(prev, dict) and isinstance(prev.get("runs"), dict):
        return prev
    return {"runs": {"legacy": prev}} if isinstance(prev, dict) and prev else {"runs": {}}


def run_key(args, name, rec_ckpt, det_ckpt) -> str:
    """Stable id for a model+config: same config -> updated, different -> appended."""
    if args.run_name:
        return args.run_name
    rec = "/".join(Path(rec_ckpt).parts[-2:])
    det = Path(det_ckpt).stem + "@" + (Path(det_ckpt).parts[-2] if len(Path(det_ckpt).parts) > 1 else "")
    return f"{name} | rec={rec} | det={det} | beams={args.num_beams} | {'joinhyph' if args.join_hyphen else 'raw'}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset folder (images + sibling <stem>.txt)")
    ap.add_argument("--name", default=None, help="dataset name (default: folder name)")
    ap.add_argument("--run-name", default=None, help="explicit key for this run (default: auto from ckpts/config)")
    ap.add_argument("--lang", default="ru")
    ap.add_argument("--det-config", type=Path, default=REPO / "detection/config.yaml")
    ap.add_argument("--det-ckpt", type=Path, default=REPO / "detection/outputs/dbnetpp_r34_hwr/best.pt")
    ap.add_argument("--rec-config", type=Path, default=REPO / "recognition/configs/finetune.yaml")
    ap.add_argument("--rec-ckpt", type=Path,
                    default=REPO / "recognition/outputs/trocr_small_bi_finetune_with_hwr200_cleaned/best")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-beams", type=int, default=4)
    ap.add_argument("--join-hyphen", action="store_true",
                    help="glue soft line-break hyphens (both ref & pred) before scoring")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=REPO / "eval_metrics_kaliningrad_essays.json",
                    help="accumulating results JSON (runs are appended, not overwritten)")
    ap.add_argument("--csv", type=Path, default=None, help="optional per-page CSV for this run")
    ap.add_argument("--save-pred", action="store_true", help="save <stem>.pred.txt next to images")
    args = ap.parse_args()

    data = Path(args.data)
    name = args.name or data.resolve().name
    pages = [p for p in sorted(data.rglob("*"))
             if p.suffix.lower() in IMG_EXT and not any(d in p.parts for d in SKIP_DIRS)
             and p.with_suffix(".txt").exists()]
    if args.limit:
        pages = pages[:args.limit]
    if not pages:
        raise SystemExit(f"no (image + .txt) pairs under {data.resolve()}")
    print(f"{len(pages)} pages with targets in {data.resolve()}")

    device = pick_device(args.device)
    det = LineDetector(args.det_config, args.det_ckpt, device)
    rec = LineRecognizer(args.rec_config, args.rec_ckpt, device, num_beams=args.num_beams)

    rows = []
    for i, img in enumerate(pages, 1):
        t0 = time.perf_counter()
        pred_text, lines = recognize_page(img, det, rec)
        dt = time.perf_counter() - t0

        ref = flat(join_lines(img.with_suffix(".txt").read_text(encoding="utf-8").splitlines(), args.join_hyphen))
        hyp = flat(join_lines(pred_text.split("\n"), args.join_hyphen))
        if not ref:
            print(f"[{i}/{len(pages)}] {img.name}  (empty target — skipped)")
            continue
        m = page_metrics(ref, hyp)
        rows.append({"page": str(img.relative_to(data)), "det_lines": len(lines),
                     "ref_chars": len(ref), "ref_words": len(ref.split()),
                     "sec": round(dt, 3), **{k: round(m[k], 4) for k in METRICS}})
        if args.save_pred:
            img.with_suffix(".pred.txt").write_text(pred_text + "\n", encoding="utf-8")
        print(f"[{i}/{len(pages)}] {img.name}  CER={m['cer']:.4f} WER={m['wer']:.4f} "
              f"NES_char={m['nes_char']:.4f} NES_word={m['nes_word']:.4f}  ({dt:.1f}s)")

    # ---- page-wise aggregate: mean + std ----
    agg = {}
    print("\n" + "=" * 58)
    print(f"{name}  |  pages={len(rows)}  (усреднение ПО СТРАНИЦАМ)")
    print("-" * 58)
    print(f"{'metric':<12}{'mean':>12}{'std':>12}")
    for k in METRICS:
        a = np.array([r[k] for r in rows], float)
        agg[k] = {"mean": round(float(a.mean()), 4), "std": round(float(a.std()), 4)}
        print(f"{k:<12}{agg[k]['mean']:>12.4f}{agg[k]['std']:>12.4f}")
    print("=" * 58)
    print("cer/wer — ниже лучше; nes_* — выше лучше. mean/std — по страницам.")

    # ---- accumulate into the results store (append, don't overwrite) ----
    key = run_key(args, name, args.rec_ckpt, args.det_ckpt)
    store = load_store(args.out)
    store["runs"][key] = {
        "dataset": name, "lang": args.lang, "samples": len(rows),
        "rec_ckpt": str(args.rec_ckpt), "det_ckpt": str(args.det_ckpt),
        "num_beams": args.num_beams, "join_hyphen": args.join_hyphen,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "metrics": agg,
        "pages": rows,                       # <- per-page results tab for this run
    }
    args.out.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved run '{key}' -> {args.out}   (runs in file: {len(store['runs'])})")

    # ---- comparison across all accumulated runs ----
    print("\n=== все прогоны в файле (сравнение) ===")
    print(f"{'run':<52}{'CER':>8}{'WER':>8}{'NESc':>8}{'NESw':>8}")
    for rk, rv in store["runs"].items():
        mt = rv.get("metrics", {})
        def g(m):
            v = mt.get(m)
            return v["mean"] if isinstance(v, dict) else (v if isinstance(v, (int, float)) else float("nan"))
        print(f"{rk[:50]:<52}{g('cer'):>8.4f}{g('wer'):>8.4f}{g('nes_char'):>8.4f}{g('nes_word'):>8.4f}")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\nper-page CSV -> {args.csv}")


if __name__ == "__main__":
    main()
