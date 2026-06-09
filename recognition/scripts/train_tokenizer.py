"""Extend the English TrOCR tokenizer with Russian tokens (Cyrillic chars + frequent words).

English ids stay unchanged; new RU tokens are appended -> model.decoder.resize_token_embeddings
keeps English rows and adds rows for the new ones.

    python scripts/train_tokenizer.py --ru-text-dirs /data/ru_texts --add-words 4000
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

RU_LETTERS = "абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
_WORD = re.compile(f"[{RU_LETTERS}]+")


def iter_txt(dirs, glob):
    for d in dirs:
        root = Path(d)
        if not root.exists():
            print(f"  warn: missing {d}"); continue
        for p in root.rglob(glob):
            if p.is_file():
                yield p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ru-text-dirs", nargs="+", required=True)
    ap.add_argument("--pretrained", default="microsoft/trocr-small-handwritten")
    ap.add_argument("--add-words", type=int, default=4000)
    ap.add_argument("--min-freq", type=int, default=3)
    ap.add_argument("--glob", default="*.txt")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "assets" / "tokenizer_bi")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.pretrained)
    old = len(tok)

    cnt = Counter()
    for p in iter_txt(args.ru_text_dirs, args.glob):
        cnt.update(_WORD.findall(p.read_text(encoding="utf-8", errors="ignore")))
    if not cnt:
        raise SystemExit("no Russian words found under --ru-text-dirs")

    vocab = set(tok.get_vocab())
    words = [w for w, c in cnt.most_common(args.add_words) if c >= args.min_freq]
    candidates = list(dict.fromkeys(list(RU_LETTERS) + words))
    added = tok.add_tokens([t for t in candidates if t not in vocab])

    args.out.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(str(args.out))
    print(f"vocab {old} -> {len(tok)}  (+{added})  -> {args.out}")
    print("RU:", tok.tokenize("пример рукописной строки 2026"))
    print("EN:", tok.tokenize("the quick brown fox"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
