#!/usr/bin/env python3
"""Same eval as kaliningrad_essays_eval.py, but prediction = YANDEX Vision OCR `fullText`.
Appends to the SAME eval_metrics_kaliningrad_essays.json so Yandex sits next to your models.

Dataset is READ-ONLY: responses are cached in ./.yandex_cache (delete it to force fresh calls).
Credentials from env: YANDEX_API_KEY, YANDEX_FOLDER_ID.

    set YANDEX_API_KEY=...   set YANDEX_FOLDER_ID=...
    python kaliningrad_essays_eval_yandex.py --data data_eval
"""
import argparse
import base64
import json
import os
import time
from pathlib import Path

import numpy as np
import jiwer
import requests

REPO = Path(__file__).resolve().parent
OUT = REPO / "eval_metrics_kaliningrad_essays.json"
CACHE = REPO / ".yandex_cache"
ENDPOINT = "https://ai.api.cloud.yandex.net/ocr/v1/recognizeText"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MIME = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".bmp": "PNG", ".tif": "PNG", ".tiff": "PNG"}


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


def yandex_fulltext(img: Path, data: Path, api_key: str, folder: str, model: str) -> str:
    cache = CACHE / (img.relative_to(data).as_posix().replace("/", "__") + ".txt")
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    body = {"mimeType": MIME.get(img.suffix.lower(), "JPEG"), "languageCodes": ["ru", "en"],
            "model": model, "content": base64.b64encode(img.read_bytes()).decode("utf-8")}
    headers = {"Content-Type": "application/json", "Authorization": f"Api-Key {api_key}",
               "x-folder-id": folder, "x-data-logging-enabled": "false"}
    r = requests.post(ENDPOINT, headers=headers, data=json.dumps(body), timeout=60)
    r.raise_for_status()
    full = r.json()["result"]["textAnnotation"].get("fullText", "") or ""
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(full, encoding="utf-8")
    time.sleep(0.2)
    return full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with images + sibling <stem>.txt (read-only)")
    ap.add_argument("--model", default="handwritten", help="handwritten | page | page-column-sort")
    args = ap.parse_args()

    api_key, folder = os.environ.get("YANDEX_API_KEY"), os.environ.get("YANDEX_FOLDER_ID")
    if not api_key or not folder:
        raise SystemExit("set env YANDEX_API_KEY and YANDEX_FOLDER_ID")

    data = Path(args.data)
    pages = [p for p in sorted(data.rglob("*"))
             if p.suffix.lower() in IMG_EXT and ".ipynb_checkpoints" not in p.parts
             and p.with_suffix(".txt").exists()]
    if not pages:
        raise SystemExit(f"no (image + .txt) pairs under {data.resolve()}")
    print(f"{len(pages)} pages in {data.resolve()}")

    cers, wers, per_page, errors = [], [], {}, 0
    for i, img in enumerate(pages, 1):
        try:
            full = yandex_fulltext(img, data, api_key, folder, args.model)
        except requests.HTTPError as e:
            errors += 1
            print(f"[{i}/{len(pages)}] {img.name}  HTTP {e.response.status_code}")
            continue
        ref = page_text(img.with_suffix(".txt").read_text(encoding="utf-8"))
        hyp = page_text(full)
        if not ref:
            continue
        cer, wer = float(jiwer.cer(ref, hyp)), float(jiwer.wer(ref, hyp))
        cers.append(cer); wers.append(wer)
        per_page[img.relative_to(data).as_posix()] = {"cer": round(cer, 4), "wer": round(wer, 4)}
        print(f"[{i}/{len(pages)}] {img.name}  CER={cer:.4f} WER={wer:.4f}")

    cer_mean, wer_mean = float(np.mean(cers)), float(np.mean(wers))
    cer_std, wer_std = float(np.std(cers)), float(np.std(wers))
    print(f"\n{data.name} | YANDEX {args.model} | pages={len(cers)}   "
          f"CER={cer_mean:.4f}±{cer_std:.4f}   WER={wer_mean:.4f}±{wer_std:.4f}")

    key = f"{data.name} | yandex | model={args.model}"
    store = load_store()
    store["runs"][key] = {"samples": len(cers), "errors": errors,
                          "cer": round(cer_mean, 4), "wer": round(wer_mean, 4),
                          "cer_std": round(cer_std, 4), "wer_std": round(wer_std, 4),
                          "per_page": per_page}
    OUT.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved -> {OUT}  (runs: {len(store['runs'])}, errors: {errors})")

    print(f"\n{'run':<52}{'CER':>8}{'±std':>8}{'WER':>8}{'±std':>8}")
    for rk, rv in store["runs"].items():
        print(f"{rk[:50]:<52}{rv.get('cer', float('nan')):>8.4f}{rv.get('cer_std', float('nan')):>8.4f}"
              f"{rv.get('wer', float('nan')):>8.4f}{rv.get('wer_std', float('nan')):>8.4f}")


if __name__ == "__main__":
    main()
