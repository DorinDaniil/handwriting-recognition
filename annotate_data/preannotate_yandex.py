#!/usr/bin/env python3
"""Pre-annotate line detection with Yandex Vision OCR, so you only touch it up by hand.

For every page image under --root that has NO <stem>.json yet (i.e. you haven't annotated it),
it calls Yandex OCR (model=handwritten), turns each recognized line into a 4-point quad + text,
and writes <stem>.json in the SAME schema the annotator (annotate.py) uses:

    {"image": "...", "width": W, "height": H, "source": "yandex",
     "lines": [{"order": 1, "polygon": [[x,y],[x,y],[x,y],[x,y]], "text": "..."}]}

Then open annotate.py — the boxes+text load automatically; you fix boxes, delete junk lines,
edit text, save. Pages you already annotated (that have a .json) are NEVER touched.

Text per line: if a sibling <stem>.txt exists AND its line count == Yandex's line count, your
existing (corrected) text is kept, paired with Yandex boxes; otherwise Yandex's own text is used
(and the page is flagged as a count mismatch to review).

    setx YANDEX_API_KEY ...   /   setx YANDEX_FOLDER_ID ...      (or set in the shell)
    pip install requests pillow
    python preannotate_yandex.py --root data
    python preannotate_yandex.py --root data --limit 5 --overwrite   # test / redo
"""
import argparse
import base64
import json
import os
import time
from pathlib import Path

import requests
from PIL import Image

ENDPOINT = "https://ai.api.cloud.yandex.net/ocr/v1/recognizeText"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MIME = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".bmp": "PNG", ".tif": "PNG", ".tiff": "PNG"}
SKIP_DIRS = {".ipynb_checkpoints", "data"}


def yandex_ocr(img_path: Path, api_key: str, folder: str, model: str) -> dict:
    content = base64.b64encode(img_path.read_bytes()).decode("utf-8")
    body = {"mimeType": MIME.get(img_path.suffix.lower(), "JPEG"),
            "languageCodes": ["ru", "en"], "model": model, "content": content}
    headers = {"Content-Type": "application/json",
               "Authorization": f"Api-Key {api_key}",
               "x-folder-id": folder,
               "x-data-logging-enabled": "false"}
    r = requests.post(ENDPOINT, headers=headers, data=json.dumps(body), timeout=60)
    r.raise_for_status()
    return r.json()


def parse_lines(resp: dict):
    """-> (api_w, api_h, [(quad, text), ...]) in reading order. quad = 4x [x,y] (TL,TR,BR,BL)."""
    ann = resp["result"]["textAnnotation"]
    out = []
    for block in ann.get("blocks", []):
        for ln in block.get("lines", []):
            xs = [int(v["x"]) for v in ln["boundingBox"]["vertices"]]
            ys = [int(v["y"]) for v in ln["boundingBox"]["vertices"]]
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
            out.append(([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], ln.get("text", "")))
    return int(ann.get("width", 0)), int(ann.get("height", 0)), out


def _txt_lines(img: Path):
    t = img.with_suffix(".txt")
    if not t.exists():
        return None
    return [ln.rstrip("\n") for ln in t.read_text(encoding="utf-8").splitlines() if ln.strip()]


def build_ann(img: Path, api_w, api_h, lines):
    W, H = Image.open(img).size
    sx = W / api_w if api_w else 1.0
    sy = H / api_h if api_h else 1.0
    existing = _txt_lines(img)
    keep_text = existing is not None and len(existing) == len(lines)   # 1:1 -> keep your text
    rec = []
    for i, (quad, ytext) in enumerate(lines):
        poly = [[int(round(x * sx)), int(round(y * sy))] for x, y in quad]
        text = existing[i] if keep_text else ytext
        rec.append({"order": i + 1, "polygon": poly, "text": text})
    return {"image": img.name, "width": W, "height": H, "source": "yandex",
            "lines": rec}, keep_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--model", default="handwritten", help="handwritten | page | page-column-sort")
    ap.add_argument("--overwrite", action="store_true", help="re-annotate even if <stem>.json exists")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.3, help="pause between requests (rate-limit)")
    ap.add_argument("--no-txt", action="store_true", help="do NOT write the per-line <stem>.txt")
    args = ap.parse_args()

    api_key = os.environ["YANDEX_API_KEY"]
    folder = os.environ["YANDEX_FOLDER_ID"]

    root = Path(args.root)

    imgs = sorted(p for p in root.rglob("*")
                  if p.suffix.lower() in IMG_EXT and not any(d in p.parts for d in SKIP_DIRS))
    done = skipped = mism = err = 0
    for img in imgs:
        jp = img.with_suffix(".json")
        if jp.exists() and not args.overwrite:
            skipped += 1
            continue
        if args.limit and done >= args.limit:
            break
        had_txt = img.with_suffix(".txt").exists()
        try:
            resp = yandex_ocr(img, api_key, folder, args.model)
            api_w, api_h, lines = parse_lines(resp)
            if not lines:
                print(f"  [no lines] {img.relative_to(root)}"); err += 1; time.sleep(args.sleep); continue
            data, keep_text = build_ann(img, api_w, api_h, lines)
            jp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            if not args.no_txt:                      # полный построчный текст рядом, 1:1 с json
                txt = "\n".join((l["text"] or "").replace("\r", " ").replace("\n", " ").replace("\t", " ")
                                for l in data["lines"])
                img.with_suffix(".txt").write_text(txt + "\n", encoding="utf-8")
            done += 1
            if keep_text:
                tag = "твой .txt подставлен 1:1"
            elif not had_txt:
                tag = "нет .txt рядом → текст Яндекса"
            else:
                tag = "строк не совпало с .txt → текст Яндекса, проверь"
                mism += 1
            print(f"  ok  {img.relative_to(root)}  lines={len(lines)}  [{tag}]")
        except requests.HTTPError as e:
            err += 1
            print(f"  HTTP {e.response.status_code} on {img.relative_to(root)}: {e.response.text[:200]}")
        except Exception as e:
            err += 1
            print(f"  ERR {img.relative_to(root)}: {e}")
        time.sleep(args.sleep)

    print(f"\ndone={done}  skipped(existing json)={skipped}  text-mismatch={mism}  errors={err}")


if __name__ == "__main__":
    main()
