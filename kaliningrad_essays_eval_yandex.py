#!/usr/bin/env python3
"""Evaluate the YANDEX Vision OCR service on the Kaliningrad essays, mirroring
kaliningrad_essays_eval.py: target = per-page <stem>.txt, prediction = Yandex `fullText`,
same metrics (cer, wer, nes_char, nes_word) averaged PER PAGE, and results ACCUMULATED into
the SAME JSON store (default eval_metrics_kaliningrad_essays.json) as another run — so Yandex
shows up next to your own models in the comparison table.

Yandex responses are cached next to each image as <stem>.yandex.txt (reused on re-runs to
avoid re-billing; pass --refresh to force new calls).

    set YANDEX_API_KEY=...   set YANDEX_FOLDER_ID=...
    python kaliningrad_essays_eval_yandex.py --data data_eval
    python kaliningrad_essays_eval_yandex.py --data data_eval --join-hyphen --csv pages_yandex.csv
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import jiwer
import requests
from rapidfuzz.distance import Levenshtein

REPO = Path(__file__).resolve().parent
ENDPOINT = "https://ai.api.cloud.yandex.net/ocr/v1/recognizeText"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MIME = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".bmp": "PNG", ".tif": "PNG", ".tiff": "PNG"}
SKIP_DIRS = {".ipynb_checkpoints"}
_WS = re.compile(r"\s+")
METRICS = ("cer", "wer", "nes_char", "nes_word")


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
    return {"cer": float(jiwer.cer(ref, hyp)),
            "wer": float(jiwer.wer(ref, hyp)),
            "nes_char": float(Levenshtein.normalized_similarity(ref, hyp)),
            "nes_word": float(Levenshtein.normalized_similarity(ref.split(), hyp.split()))}


def load_store(path: Path) -> dict:
    if not path.exists():
        return {"runs": {}}
    try:
        prev = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"runs": {}}
    if isinstance(prev, dict) and isinstance(prev.get("runs"), dict):
        return prev
    return {"runs": {"legacy": prev}} if isinstance(prev, dict) and prev else {"runs": {}}


def yandex_fulltext(img: Path, api_key: str, folder: str, model: str, refresh: bool) -> str:
    """Return Yandex page `fullText`, caching to <stem>.yandex.txt (reused unless --refresh)."""
    cache = img.with_suffix(".yandex.txt")
    if cache.exists() and not refresh:
        return cache.read_text(encoding="utf-8")
    body = {"mimeType": MIME.get(img.suffix.lower(), "JPEG"), "languageCodes": ["ru", "en"],
            "model": model, "content": base64.b64encode(img.read_bytes()).decode("utf-8")}
    headers = {"Content-Type": "application/json", "Authorization": f"Api-Key {api_key}",
               "x-folder-id": folder, "x-data-logging-enabled": "false"}
    r = requests.post(ENDPOINT, headers=headers, data=json.dumps(body), timeout=60)
    r.raise_for_status()
    full = r.json()["result"]["textAnnotation"].get("fullText", "") or ""
    cache.write_text(full, encoding="utf-8")
    return full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset folder (images + sibling <stem>.txt)")
    ap.add_argument("--name", default=None, help="dataset name (default: folder name)")
    ap.add_argument("--run-name", default=None, help="explicit key for this run")
    ap.add_argument("--lang", default="ru")
    ap.add_argument("--model", default="handwritten", help="handwritten | page | page-column-sort")
    ap.add_argument("--api-key", default=os.environ.get("YANDEX_API_KEY"))
    ap.add_argument("--folder", default=os.environ.get("YANDEX_FOLDER_ID"))
    ap.add_argument("--join-hyphen", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.2, help="pause between API calls")
    ap.add_argument("--refresh", action="store_true", help="ignore cache, re-call Yandex")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=REPO / "eval_metrics_kaliningrad_essays.json",
                    help="accumulating results JSON (same file as the pipeline eval)")
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()

    if not args.api_key or not args.folder:
        raise SystemExit("set YANDEX_API_KEY / YANDEX_FOLDER_ID (env) or pass --api-key/--folder")

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

    rows, errors = [], 0
    for i, img in enumerate(pages, 1):
        try:
            t0 = time.perf_counter()
            full = yandex_fulltext(img, args.api_key, args.folder, args.model, args.refresh)
            dt = time.perf_counter() - t0
        except requests.HTTPError as e:
            errors += 1
            print(f"[{i}/{len(pages)}] {img.name}  HTTP {e.response.status_code}: {e.response.text[:120]}")
            continue

        ref = flat(join_lines(img.with_suffix(".txt").read_text(encoding="utf-8").splitlines(), args.join_hyphen))
        hyp = flat(join_lines(full.split("\n"), args.join_hyphen))
        if not ref:
            print(f"[{i}/{len(pages)}] {img.name}  (empty target — skipped)")
            continue
        m = page_metrics(ref, hyp)
        rows.append({"page": str(img.relative_to(data)), "ref_chars": len(ref), "ref_words": len(ref.split()),
                     "sec": round(dt, 3), **{k: round(m[k], 4) for k in METRICS}})
        print(f"[{i}/{len(pages)}] {img.name}  CER={m['cer']:.4f} WER={m['wer']:.4f} "
              f"NES_char={m['nes_char']:.4f} NES_word={m['nes_word']:.4f}")
        if not (img.with_suffix(".yandex.txt").exists() and not args.refresh):
            time.sleep(args.sleep)

    agg = {}
    print("\n" + "=" * 58)
    print(f"{name} | YANDEX {args.model}  |  pages={len(rows)}  (усреднение ПО СТРАНИЦАМ)")
    print("-" * 58)
    print(f"{'metric':<12}{'mean':>12}{'std':>12}")
    for k in METRICS:
        a = np.array([r[k] for r in rows], float)
        agg[k] = {"mean": round(float(a.mean()), 4), "std": round(float(a.std()), 4)}
        print(f"{k:<12}{agg[k]['mean']:>12.4f}{agg[k]['std']:>12.4f}")
    print("=" * 58)

    key = args.run_name or f"{name} | engine=yandex | model={args.model} | {'joinhyph' if args.join_hyphen else 'raw'}"
    store = load_store(args.out)
    store["runs"][key] = {
        "dataset": name, "lang": args.lang, "engine": "yandex", "model": args.model,
        "samples": len(rows), "errors": errors, "join_hyphen": args.join_hyphen,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "metrics": agg, "pages": rows,
    }
    args.out.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved run '{key}' -> {args.out}   (runs in file: {len(store['runs'])}, errors: {errors})")

    print("\n=== все прогоны в файле (сравнение) ===")
    print(f"{'run':<52}{'CER':>8}{'WER':>8}{'NESc':>8}{'NESw':>8}")
    for rk, rv in store["runs"].items():
        mt = rv.get("metrics", {})
        def g(m):
            v = mt.get(m)
            return v["mean"] if isinstance(v, dict) else (v if isinstance(v, (int, float)) else float("nan"))
        print(f"{rk[:50]:<52}{g('cer'):>8.4f}{g('wer'):>8.4f}{g('nes_char'):>8.4f}{g('nes_word'):>8.4f}")

    if args.csv and rows:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\nper-page CSV -> {args.csv}")


if __name__ == "__main__":
    main()
