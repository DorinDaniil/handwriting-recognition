#!/usr/bin/env python3
"""Filter IMGUR5K word crops with Qwen3-VL into a JSON/TSV of good handwriting.

Keeps only words that are:
  - handwritten on paper/notebook (NOT graffiti, walls, signs, tattoos, screens, street scenes),
  - roughly horizontal (NOT rotated/vertical/strong angle),
  - fully inside the crop (NOT cut off),
  - a single legible word.
A cheap text pre-filter drops single letters and non-Latin / accented (umlaut) words before
the VLM ever runs. Every decision is logged to a .jsonl so the run resumes after interruption.

    python scripts/filter_imgur5k.py --tsv data/imgur5k/train.tsv
    python scripts/filter_imgur5k.py --limit 300            # quick sanity run
    python scripts/filter_imgur5k.py --model Qwen/Qwen3-VL-4B-Instruct   # faster
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

JUDGE = (
    "You see one cropped image that should be a single handwritten word. "
    "Answer KEEP only if ALL of these hold:\n"
    "- handwritten on paper or in a notebook (NOT graffiti, a wall, a sign, a tattoo, a screen, "
    "or an outdoor/street scene);\n"
    "- the word is roughly horizontal (NOT vertical, rotated, or at a strong angle);\n"
    "- the whole word is fully inside the crop (NOT cut off at the edges);\n"
    "- it is one clear, legible word.\n"
    "Otherwise answer DROP. Reply with exactly one word: KEEP or DROP."
)

_WORD = re.compile(r"[A-Za-z0-9'\-.,!?]+")


def text_ok(w: str) -> bool:
    """Cheap, VLM-free reject: single letters, non-Latin, accented/umlaut, digits-only."""
    w = w.strip()
    if len(w) < 2:
        return False
    if not _WORD.fullmatch(w):                  # ASCII Latin only -> drops umlauts/cyrillic/etc.
        return False
    return sum(c.isalpha() for c in w) >= 2


def load_model(model_id, device, dtype):
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=dtype, device_map="auto" if device == "cuda" else None)
    if device == "cpu":
        model.to(device)
    model.eval()
    print("loaded", model_id)
    return processor, model


@torch.no_grad()
def keep_image(processor, model, image, min_h=64) -> bool:
    if image.height < min_h:
        s = min_h / image.height
        image = image.resize((max(1, int(image.width * s)), min_h))
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": JUDGE}]}]
    inp = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                        return_dict=True, return_tensors="pt").to(model.device)
    out = model.generate(**inp, max_new_tokens=4, do_sample=False)
    ans = processor.decode(out[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True).upper()
    return "KEEP" in ans and "DROP" not in ans


def _write_split(tsv: Path, pairs, decisions, out_dir: Path) -> int:
    """Write <split>_good.json / <split>_good.tsv for the kept pairs of one split."""
    good = [{"path": p, "word": w} for p, w in pairs if decisions.get(p, ("", False))[1]]
    (out_dir / f"{tsv.stem}_good.json").write_text(
        json.dumps(good, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / f"{tsv.stem}_good.tsv").write_text(
        "".join(f"{g['path']}\t{g['word']}\n" for g in good), encoding="utf-8")
    return len(good)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", nargs="+", default=["data/imgur5k/train.tsv", "data/imgur5k/test.tsv"],
                    help="one or more crop_path<TAB>word files (default: train + test)")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--min-h", type=int, default=64, help="upscale thin crops to this height")
    ap.add_argument("--limit", type=int, default=0, help="cap total pairs (sanity run)")
    ap.add_argument("--save-every", type=int, default=500)
    args = ap.parse_args()

    tsvs = [Path(t) if Path(t).is_absolute() else ROOT / t for t in args.tsv]
    tsvs = [t for t in tsvs if t.exists()]
    if not tsvs:
        raise SystemExit("no input tsv found")
    out_dir = tsvs[0].parent
    log = out_dir / "imgur5k_filter_decisions.jsonl"

    split_pairs = {t: [tuple(ln.split("\t", 1)) for ln in t.read_text(encoding="utf-8").splitlines() if "\t" in ln]
                   for t in tsvs}
    all_pairs = [pw for pairs in split_pairs.values() for pw in pairs]
    if args.limit:
        all_pairs = all_pairs[:args.limit]

    decisions = {}                                   # path -> (word, keep) — shared resume state
    if log.exists():
        for ln in log.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                d = json.loads(ln)
                decisions[d["path"]] = (d["word"], d["keep"])
    todo = [(p, w) for p, w in all_pairs if p not in decisions]
    print(f"splits: {[t.name for t in tsvs]} | pairs: {len(all_pairs)} | decided: {len(decisions)} | to do: {len(todo)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    processor = model = None                          # lazy: load VLM only if something passes text_ok

    with open(log, "a", encoding="utf-8") as logf:
        for i, (path, word) in enumerate(tqdm(todo)):
            keep = False
            if text_ok(word):
                try:
                    img = Image.open(path).convert("RGB")
                except Exception:
                    img = None
                if img is not None:
                    if processor is None:
                        processor, model = load_model(args.model, device, dtype)
                    keep = keep_image(processor, model, img, args.min_h)
            decisions[path] = (word, keep)
            logf.write(json.dumps({"path": path, "word": word, "keep": keep}, ensure_ascii=False) + "\n")
            logf.flush()
            if (i + 1) % args.save_every == 0:
                for t, pairs in split_pairs.items():
                    _write_split(t, pairs, decisions, out_dir)

    for t, pairs in split_pairs.items():
        n = _write_split(t, pairs, decisions, out_dir)
        print(f"{t.stem}: good {n} / {len(pairs)}  ->  {out_dir / (t.stem + '_good.tsv')}")


if __name__ == "__main__":
    main()
