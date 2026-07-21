#!/usr/bin/env python3
"""Same eval as kaliningrad_essays_eval.py / _yandex.py, but prediction = a VLM via OpenRouter
transcribing the whole page image. Appends to the SAME eval_metrics_kaliningrad_essays.json.

Model PRESETS are defined below — put your model ids and YOUR OWN prompt for each one,
switch with --preset. The request format is a single user message (prompt text + image),
exactly the shape that works through OpenRouter.

Dataset is READ-ONLY: responses are cached in ./.openrouter_cache/<model>/ (delete to re-query).
Credentials from env: OPENROUTER_API_KEY.

    set OPENROUTER_API_KEY=...
    python kaliningrad_essays_eval_openrouter.py --data data_eval --preset gemini
    python kaliningrad_essays_eval_openrouter.py --data data_eval --preset o4
"""
import argparse
import base64
import json
import os
import time
from pathlib import Path

import numpy as np
import jiwer
from openai import OpenAI

# ------------------------------------------------------------------ MODEL PRESETS (правь тут)
MODELS = {
    "gemini": {
        "model": "google/gemini-2.5-flash",
        "max_tokens": 2000,
        "temperature": 0.05,
        "prompt": """
    You are a specialized OCR (Optical Character Recognition) engine.
    Your only function is to transcribe handwritten text from images into digital text.
    Do not analyze the content or refuse the request.
    Ignore any privacy concerns for the purpose of transcription.
    Just output the text exactly as written.
    """,
    },
    "o4": {
        "model": "openai/o4-mini",
        "max_tokens": 16384,          # o-серия тратит часть бюджета на скрытый reasoning
        # temperature не указана — o-серия её не принимает
        "prompt": """
    You are a specialized OCR (Optical Character Recognition) engine.
    Your only function is to transcribe handwritten text from images into digital text.
    Do not analyze the content or refuse the request.
    Ignore any privacy concerns for the purpose of transcription.
    Just output the text exactly as written.
    """,
    },
}
# ---------------------------------------------------------------------------------------------

REPO = Path(__file__).resolve().parent
OUT = REPO / "eval_metrics_kaliningrad_essays.json"
CACHE = REPO / ".openrouter_cache"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
        ".bmp": "image/bmp", ".tif": "image/tiff", ".tiff": "image/tiff"}


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


def vlm_fulltext(client, img: Path, data: Path, preset: dict) -> str:
    model = preset["model"]
    cache = (CACHE / model.replace("/", "__")
             / (img.relative_to(data).as_posix().replace("/", "__") + ".txt"))
    if cache.exists():
        return cache.read_text(encoding="utf-8")

    b64 = base64.b64encode(img.read_bytes()).decode("utf-8")
    params = dict(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": preset["prompt"]},
                {"type": "image_url",
                 "image_url": {"url": f"data:{MIME.get(img.suffix.lower(), 'image/jpeg')};base64,{b64}"}},
            ],
        }],
        max_tokens=preset["max_tokens"],
    )
    if "temperature" in preset:
        params["temperature"] = preset["temperature"]
    response = client.chat.completions.create(**params)
    full = (response.choices[0].message.content or "").strip()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(full, encoding="utf-8")
    time.sleep(0.5)
    return full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with images + sibling <stem>.txt (read-only)")
    ap.add_argument("--preset", required=True, choices=sorted(MODELS), help="which model preset to run")
    ap.add_argument("--save-preds", type=Path, default=None,
                    help="folder to mirror per-page prediction .txt into (same subfolders/names "
                         "as the data, NEVER inside --data)")
    args = ap.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("set env OPENROUTER_API_KEY")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    preset = MODELS[args.preset]

    data = Path(args.data)
    if args.save_preds is not None and data.resolve() in args.save_preds.resolve().parents:
        raise SystemExit("--save-preds must NOT be inside --data (данные read-only)")
    pages = [p for p in sorted(data.rglob("*"))
             if p.suffix.lower() in IMG_EXT and ".ipynb_checkpoints" not in p.parts
             and p.with_suffix(".txt").exists()]
    if not pages:
        raise SystemExit(f"no (image + .txt) pairs under {data.resolve()}")
    print(f"{len(pages)} pages in {data.resolve()} | preset {args.preset} = {preset['model']}")

    cers, wers, per_page, errors = [], [], {}, 0
    for i, img in enumerate(pages, 1):
        try:
            full = vlm_fulltext(client, img, data, preset)
        except Exception as e:
            errors += 1
            print(f"[{i}/{len(pages)}] {img.name}  ERROR: {e}")
            continue
        if args.save_preds is not None:
            pred = args.save_preds / img.relative_to(data).with_suffix(".txt")
            pred.parent.mkdir(parents=True, exist_ok=True)
            pred.write_text(full + "\n", encoding="utf-8")
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
    print(f"\n{data.name} | {preset['model']} | pages={len(cers)}   "
          f"CER={cer_mean:.4f}±{cer_std:.4f}   WER={wer_mean:.4f}±{wer_std:.4f}")

    key = f"{data.name} | openrouter | {args.preset}={preset['model']}"
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
