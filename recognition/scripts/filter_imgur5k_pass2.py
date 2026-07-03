#!/usr/bin/env python3
"""Second, stricter pass over the IMGUR5K *_good.tsv files.

The first pass (filter_imgur5k.py) still lets through some rotated / strongly-angled words
and words cut off at the crop edges. This pass re-judges ONLY the already-kept pairs (so it
is fast and needs no text pre-filter) with a prompt focused purely on orientation and
completeness, and writes <stem>_v2.json / <stem>_v2.tsv.

    python scripts/filter_imgur5k_pass2.py                                   # train_good + test_good
    python scripts/filter_imgur5k_pass2.py --tsv data/imgur5k/train_good.tsv
    python scripts/filter_imgur5k_pass2.py --model Qwen/Qwen3-VL-4B-Instruct # faster
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

JUDGE = (
    "You see a cropped image of a single handwritten word. "
    "Answer KEEP only if BOTH hold:\n"
    "- the word is horizontal (NOT vertical, NOT rotated 90 degrees, NOT tilted at a strong "
    "angle — a slight natural slant is fine);\n"
    "- the whole word is fully visible, with NO letters cut off at the left, right, top or "
    "bottom edge of the crop.\n"
    "Otherwise answer DROP. Reply with exactly one word: KEEP or DROP."
)


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
    good = [{"path": p, "word": w} for p, w in pairs if decisions.get(p, ("", False))[1]]
    stem = tsv.stem + "_v2"
    (out_dir / f"{stem}.json").write_text(
        json.dumps(good, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / f"{stem}.tsv").write_text(
        "".join(f"{g['path']}\t{g['word']}\n" for g in good), encoding="utf-8")
    return len(good)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", nargs="+",
                    default=["data/imgur5k/train_good.tsv", "data/imgur5k/test_good.tsv"],
                    help="already-filtered *_good.tsv files (default: train + test)")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--min-h", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=500)
    args = ap.parse_args()

    tsvs = [Path(t) if Path(t).is_absolute() else ROOT / t for t in args.tsv]
    tsvs = [t for t in tsvs if t.exists()]
    if not tsvs:
        raise SystemExit("no input *_good.tsv found (run filter_imgur5k.py first)")
    out_dir = tsvs[0].parent
    log = out_dir / "imgur5k_filter_pass2_decisions.jsonl"

    split_pairs = {t: [tuple(ln.split("\t", 1)) for ln in t.read_text(encoding="utf-8").splitlines() if "\t" in ln]
                   for t in tsvs}
    all_pairs = [pw for pairs in split_pairs.values() for pw in pairs]
    if args.limit:
        all_pairs = all_pairs[:args.limit]

    decisions = {}
    if log.exists():
        for ln in log.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                d = json.loads(ln)
                decisions[d["path"]] = (d["word"], d["keep"])
    todo = [(p, w) for p, w in all_pairs if p not in decisions]
    print(f"splits: {[t.name for t in tsvs]} | pairs: {len(all_pairs)} | decided: {len(decisions)} | to do: {len(todo)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    processor = model = None

    with open(log, "a", encoding="utf-8") as logf:
        for i, (path, word) in enumerate(tqdm(todo)):
            keep = False
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
        print(f"{t.stem}: good {n} / {len(pairs)}  ->  {out_dir / (t.stem + '_v2.tsv')}")


if __name__ == "__main__":
    main()
