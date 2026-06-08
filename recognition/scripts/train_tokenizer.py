"""Train a compact Russian (Cyrillic) BPE tokenizer for TrOCR-small.

Trains a byte-level BPE on your .txt corpus (the same folders you feed the
generator) and saves a HuggingFace-loadable tokenizer with TrOCR's RoBERTa-style
special tokens (<s> <pad> </s> <unk> <mask>). Byte-level BPE covers all Cyrillic.

    python scripts/train_tokenizer.py --text-dirs /mnt/.../data/texts --vocab-size 4000

Load it later with:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("assets/tokenizer_ru")

Tip: a small vocab (2000-8000) is plenty for handwriting and keeps the decoder
embedding matrix light. For the leanest, most robust option you can also skip
training and use a char-level vocab, or keep the English byte-BPE (then every
weight loads, see src/model.build_trocr_small).
"""
from __future__ import annotations

import argparse
from pathlib import Path

SPECIAL = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]


def iter_txt(dirs, glob: str):
    for d in dirs:
        root = Path(d)
        if not root.exists():
            print(f"  warn: missing dir {d}")
            continue
        for p in root.rglob(glob):
            if p.is_file():
                yield str(p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-dirs", nargs="+", required=True, help="folders with .txt")
    ap.add_argument("--vocab-size", type=int, default=4000)
    ap.add_argument("--min-frequency", type=int, default=2)
    ap.add_argument("--glob", default="*.txt")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[1] / "assets" / "tokenizer_ru")
    args = ap.parse_args()

    files = list(iter_txt(args.text_dirs, args.glob))
    if not files:
        raise SystemExit("no .txt files found under --text-dirs")
    print(f"training on {len(files)} files -> vocab {args.vocab_size}")

    from tokenizers import ByteLevelBPETokenizer

    bpe = ByteLevelBPETokenizer()
    bpe.train(files=files, vocab_size=args.vocab_size,
              min_frequency=args.min_frequency, special_tokens=SPECIAL)
    args.out.mkdir(parents=True, exist_ok=True)
    bpe.save_model(str(args.out))

    # wrap as a HF fast tokenizer (RoBERTa-style) so AutoTokenizer can load it
    from transformers import RobertaTokenizerFast

    fast = RobertaTokenizerFast(
        vocab_file=str(args.out / "vocab.json"),
        merges_file=str(args.out / "merges.txt"),
        bos_token="<s>", eos_token="</s>", pad_token="<pad>",
        unk_token="<unk>", mask_token="<mask>",
    )
    fast.save_pretrained(str(args.out))
    print(f"tokenizer -> {args.out}  (vocab {fast.vocab_size})")
    sample = "пример рукописной строки 2026"
    print("roundtrip:", repr(fast.decode(fast(sample).input_ids, skip_special_tokens=True)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
