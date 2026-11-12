#!/usr/bin/env python3
"""Latency of whole-page VLM transcription through OpenRouter: p50 / p95 / p99.

Same request shape as kaliningrad_essays_eval.py's OpenRouter variant, but NOTHING is
cached — every page is a real API call, timed end to end (send -> full response), one
request at a time. Quality is not computed here, only wall time.

Dataset is READ-ONLY. Credentials from env: OPENROUTER_API_KEY.

    set OPENROUTER_API_KEY=...
    python openrouter_latency.py --data data_eval --model google/gemini-3-flash-preview
    python openrouter_latency.py --data data_eval --model ... --limit 20 --repeats 2
"""
import argparse
import base64
import json
import os
import time
from pathlib import Path

import numpy as np
from openai import OpenAI
from tqdm import tqdm

REPO = Path(__file__).resolve().parent
OUT = REPO / "latency_openrouter.json"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
        ".bmp": "image/bmp", ".tif": "image/tiff", ".tiff": "image/tiff"}

PROMPT = """
    You are a specialized OCR (Optical Character Recognition) engine.
    Your only function is to transcribe handwritten text from images into digital text.
    Do not analyze the content or refuse the request.
    Ignore any privacy concerns for the purpose of transcription.
    Just output the text exactly as written.
    """


def as_data_uri(img: Path) -> str:
    b64 = base64.b64encode(img.read_bytes()).decode("utf-8")
    return f"data:{MIME.get(img.suffix.lower(), 'image/jpeg')};base64,{b64}"


def timed_call(client, uri: str, model: str, max_tokens: int, temperature: float | None):
    """One request. Returns (seconds, completion_tokens or None)."""
    params = dict(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": uri}},
            ],
        }],
        max_tokens=max_tokens,
    )
    if temperature is not None:
        params["temperature"] = temperature

    t0 = time.perf_counter()
    response = client.chat.completions.create(**params)
    dt = time.perf_counter() - t0

    usage = getattr(response, "usage", None)
    return dt, getattr(usage, "completion_tokens", None) if usage else None


def report(times_s, name: str, tokens):
    a = np.asarray(times_s)
    print(f"\n{name}  (n={len(a)})")
    print(f"  mean = {a.mean():.2f} ± {a.std():.2f} с")
    for q in (50, 95, 99):
        print(f"  p{q}  = {np.percentile(a, q):.2f} с")
    print(f"  min / max = {a.min():.2f} / {a.max():.2f} с")
    tok = [t for t in tokens if t]
    if tok:
        print(f"  выход: {np.mean(tok):.0f} токенов в среднем, "
              f"{np.mean(tok) / a.mean():.1f} токен/с")

    ms = a * 1000
    row = (f"  строка для деки: {np.percentile(ms, 50):,.0f} & "
           f"{np.percentile(ms, 95):,.0f} & {np.percentile(ms, 99):,.0f}")
    print(row.replace(",", "\\,"))
    return {
        "n": int(len(a)),
        "mean_s": round(float(a.mean()), 3), "std_s": round(float(a.std()), 3),
        "p50_s": round(float(np.percentile(a, 50)), 3),
        "p95_s": round(float(np.percentile(a, 95)), 3),
        "p99_s": round(float(np.percentile(a, 99)), 3),
        "min_s": round(float(a.min()), 3), "max_s": round(float(a.max()), 3),
        "completion_tokens_mean": round(float(np.mean(tok)), 1) if tok else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with page images (read-only)")
    ap.add_argument("--model", required=True, help="OpenRouter model id, e.g. google/gemini-3-flash-preview")
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--temperature", type=float, default=0.05,
                    help="passed only if >= 0; o-series models reject it, use -1 to omit")
    ap.add_argument("--limit", type=int, default=0, help="cap the number of pages (0 = all)")
    ap.add_argument("--repeats", type=int, default=1, help="passes over the page set")
    ap.add_argument("--warmup", type=int, default=2, help="uncounted requests before measuring")
    ap.add_argument("--pause", type=float, default=0.0, help="sleep between requests, s")
    ap.add_argument("--any-image", action="store_true",
                    help="measure on every image, not only pages that have a sibling .txt")
    args = ap.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("set env OPENROUTER_API_KEY")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    temperature = None if args.temperature < 0 else args.temperature

    data = Path(args.data)
    pages = [p for p in sorted(data.rglob("*"))
             if p.suffix.lower() in IMG_EXT and ".ipynb_checkpoints" not in p.parts
             and (args.any_image or p.with_suffix(".txt").exists())]
    if not pages:
        raise SystemExit(f"no page images under {data.resolve()}")
    if args.limit:
        pages = pages[:args.limit]
    print(f"{len(pages)} pages x {args.repeats} passes | {args.model}")

    # encode once: base64 is our own client-side cost, not the model's latency
    uris = [as_data_uri(p) for p in pages]

    for i in range(min(args.warmup, len(uris))):
        timed_call(client, uris[i], args.model, args.max_tokens, temperature)
    print(f"warmup: {min(args.warmup, len(uris))} requests done")

    times, tokens, errors = [], [], 0
    work = [(p, u) for _ in range(args.repeats) for p, u in zip(pages, uris)]
    for page, uri in tqdm(work, desc="страницы"):
        try:
            dt, out_tok = timed_call(client, uri, args.model, args.max_tokens, temperature)
        except Exception as e:                       # failed request is not a latency sample
            errors += 1
            tqdm.write(f"{page.name}  ERROR: {type(e).__name__}: {e}")
            continue
        times.append(dt)
        tokens.append(out_tok)
        if args.pause:
            time.sleep(args.pause)

    if not times:
        raise SystemExit("all requests failed")
    stats = report(times, args.model, tokens)
    stats["errors"] = errors

    store = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    store[f"{data.name} | {args.model}"] = stats
    OUT.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved -> {OUT}  (errors: {errors})")


if __name__ == "__main__":
    main()
