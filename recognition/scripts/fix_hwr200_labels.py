#!/usr/bin/env python3
"""Clean hwr200 label files: remove '[' and ']' brackets and collapse the spaces they
leave behind (so "слово [ ] слово" -> "слово слово"). Originals are NOT touched — fixed
labels are written to a NEW folder (default: <input>_fixed). Prints the total bracket count.

    python scripts/fix_hwr200_labels.py /mnt/.../hwr200/filtered_lines_v2/lines_txt
    python scripts/fix_hwr200_labels.py /mnt/.../lines_txt --out /mnt/.../lines_txt_clean
    python scripts/fix_hwr200_labels.py /mnt/.../lines_txt --dry-run   # only count
"""
import argparse
import re
from pathlib import Path


def clean(text: str) -> tuple[str, int]:
    n = text.count("[") + text.count("]")
    text = text.replace("[", "").replace("]", "")
    text = re.sub(r"[ \t]+", " ", text)                       # collapse spaces left by removal
    text = "\n".join(line.strip() for line in text.splitlines())
    return text.strip(), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="labels dir (e.g. .../lines_txt) or a single .txt")
    ap.add_argument("--out", default=None,
                    help="output dir for fixed labels (default: <input>_fixed). Originals untouched.")
    ap.add_argument("--dry-run", action="store_true", help="count only, write nothing")
    args = ap.parse_args()

    root = Path(args.path)
    if root.is_file():
        files, base = [root], root.parent
    else:
        files, base = sorted(root.rglob("*.txt")), root
    out_dir = Path(args.out) if args.out else base.parent / f"{base.name}_fixed"

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    total, changed = 0, 0
    for f in files:
        cleaned, n = clean(f.read_text(encoding="utf-8"))
        total += n
        changed += n > 0
        if not args.dry_run:
            dst = out_dir / f.relative_to(base)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(cleaned, encoding="utf-8")

    print(f"files: {len(files)} | with brackets: {changed} | brackets found: {total}")
    if not args.dry_run:
        print(f"fixed labels -> {out_dir}")


if __name__ == "__main__":
    main()
