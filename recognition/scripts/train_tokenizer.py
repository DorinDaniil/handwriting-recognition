"""Train a fresh byte-level BPE (EN+RU) for TrOCR.

The trocr-small base tokenizer is a Unigram model, and retraining it
(train_new_from_iterator) runs the Unigram trainer, which is fragile (NAN / "sentence
too long"). Instead we train a robust byte-level BPE from scratch on your EN+RU corpus.
Byte-level decode reconstructs spaces correctly (unlike add_tokens), so encode->decode
round-trips for both languages; Russian stays compact. The tokenizer type is independent
of the model — build_trocr_small re-initialises the decoder embeddings for the new vocab.

    python scripts/train_tokenizer.py \
        --ru-text-dirs /data/ru1 /data/ru2 --en-text-dirs /data/en --vocab-size 32000
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

SPECIAL = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]


def _files(dirs, glob, cap):
    out = []
    for d in dirs:
        root = Path(d)
        if not root.exists():
            print("  warn: missing", d); continue
        out += [p for p in root.rglob(glob) if p.is_file()]
    random.shuffle(out)
    return out[:cap] if cap else out


def _lines(files, max_chars):
    for p in files:
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for ln in txt.splitlines():
            ln = ln.strip()
            if ln:
                yield ln[:max_chars]          # cap to keep the trainer stable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ru-text-dirs", nargs="*", default=[])
    ap.add_argument("--en-text-dirs", nargs="*", default=[])
    ap.add_argument("--vocab-size", type=int, default=65000)
    ap.add_argument("--min-frequency", type=int, default=2)
    ap.add_argument("--glob", default="*.txt")
    ap.add_argument("--max-files-per-lang", type=int, default=350000, help="sample files for speed")
    ap.add_argument("--max-line-chars", type=int, default=2000, help="truncate very long lines")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[1] / "assets" / "tokenizer_bi")
    args = ap.parse_args()

    ru = _files(args.ru_text_dirs, args.glob, args.max_files_per_lang)
    en = _files(args.en_text_dirs, args.glob, args.max_files_per_lang)
    files = ru + en
    random.shuffle(files)
    if not files:
        raise SystemExit("no .txt found; pass --ru-text-dirs / --en-text-dirs")
    print(f"training byte-level BPE on {len(ru)} ru + {len(en)} en files, vocab={args.vocab_size}")

    from tokenizers import ByteLevelBPETokenizer
    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator(_lines(files, args.max_line_chars), vocab_size=args.vocab_size,
                            min_frequency=args.min_frequency, special_tokens=SPECIAL)
    args.out.mkdir(parents=True, exist_ok=True)
    bpe.save_model(str(args.out))

    from transformers import RobertaTokenizerFast
    tok = RobertaTokenizerFast(
        vocab_file=str(args.out / "vocab.json"), merges_file=str(args.out / "merges.txt"),
        bos_token="<s>", eos_token="</s>", pad_token="<pad>", unk_token="<unk>", mask_token="<mask>")
    tok.save_pretrained(str(args.out))
    print(f"vocab {len(tok)}  -> {args.out}")
    for s in ("оказываются сильнее его стремлений", "the quick brown fox jumps over"):
        rt = tok.decode(tok(s).input_ids, skip_special_tokens=True)
        print("roundtrip:", repr(rt), "OK" if rt == s else "!! MISMATCH")


if __name__ == "__main__":
    main()
