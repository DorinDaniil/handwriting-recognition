#!/usr/bin/env python3
"""Download the CVL handwriting database (English + German) and build a line-level
manifest (image_path<TAB>transcript) compatible with src.finetune.TsvLineDataset.

CVL is free for research but you must accept the terms on the CVL site; verify the URL:
    https://cvl.tuwien.ac.at/research/cvl-databases/

    python scripts/download/cvl.py --root data/cvl --preview
    python scripts/download/cvl.py --root data/cvl
"""
import argparse
import difflib
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DEFAULT_URL = "https://zenodo.org/records/1492267/files/cvl-database-1-1.zip"
TEXT_ATTR = "text"
IMG_EXT = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")
_DECL = re.compile(r"^\s*<\?xml.*?\?>", re.DOTALL)


def _strip_ns(tag):
    return tag.rsplit("}", 1)[-1]


def _text_of(el):
    for key, value in el.attrib.items():
        if _strip_ns(key).lower() == TEXT_ATTR:
            return value
    return None


def _decode(raw: bytes) -> str:
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")              # BOM -> correct endianness, BOM consumed
    if raw[:3] == b"\xef\xbb\xbf":
        return raw.decode("utf-8-sig")
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")


def _parse_xml(path: Path):
    text = _decode(path.read_bytes()).lstrip("﻿ \t\r\n")   # drop any residual BOM / leading ws
    return ET.fromstring(_DECL.sub("", text, count=1))          # drop decl: ET rejects it on a str


def download_archive(url, dst: Path):
    if dst.exists():
        print(f"archive present -> {dst}")
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    with urllib.request.urlopen(url) as r, open(dst, "wb") as f:
        total, done = int(r.headers.get("Content-Length", 0)), 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:6.0f} / {total / 1e6:.0f} MB", end="")
    print()
    return dst


def extract(zip_path: Path, root: Path):
    if not any(root.glob("**/xml")):
        print(f"extracting -> {root}")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(root)
    return root


def _word_box(el):
    xs, ys = [], []
    for p in el.iter():
        if _strip_ns(p.tag) == "Point":
            xs.append(float(p.get("x", 0)))
            ys.append(float(p.get("y", 0)))
    return (min(xs), min(ys), max(xs), max(ys)) if xs else None


def _image_filename(root_el):
    for el in root_el.iter():
        if _strip_ns(el.tag) == "Page" and el.get("imageFilename"):
            return el.get("imageFilename")
    return None


def _lines(root_el):
    parent = {c: p for p in root_el.iter() for c in p}
    items = [(el, _text_of(el).strip(), _word_box(el)) for el in root_el.iter()
             if _text_of(el) and _text_of(el).strip() and _word_box(el)]   # word = text + Points
    groups = {}
    for el, text, box in items:
        groups.setdefault(id(parent.get(el)), []).append((text, box))      # group words by line
    lines = []
    for grp in groups.values():
        grp.sort(key=lambda tb: tb[1][0])                                  # left -> right
        words = [t for t, _ in grp]
        if not words:
            continue
        x0 = min(b[0] for _, b in grp)
        y0 = min(b[1] for _, b in grp)
        x1 = max(b[2] for _, b in grp)
        y1 = max(b[3] for _, b in grp)
        lines.append((y0, (x0, y0, x1, y1), words))
    lines.sort(key=lambda line: line[0])                                   # top -> bottom
    return [(box, words) for _, box, words in lines]


_WORD_RE = re.compile(r"[A-Za-z0-9’']+")


def _space_dashes(text):
    return re.sub(r"\s+", " ", re.sub(r"([—–])", r" \1 ", text)).strip()   # dashes get surrounding spaces


def _ref_tokens(ref):
    words = list(_WORD_RE.finditer(ref))
    tokens = []
    for i, m in enumerate(words):
        nxt = words[i + 1].start() if i + 1 < len(words) else len(ref)
        suffix = re.sub(r"\s+", "", ref[m.end():nxt])      # punctuation after the word (spaces dropped)
        tokens.append((m.group(), suffix))
    return tokens


def recover(lines, references):
    """Match each CVL line on its own to the best-fitting reference (by word coverage) and
    inject that reference's punctuation onto the line's own words. Robust to per-writer line
    wrapping. A line matching no reference well enough stays words-only."""
    ref_tokens = [_ref_tokens(ref) for ref in references]
    out = []
    for box, words in lines:
        if not words:
            out.append((box, ""))
            continue
        bare = [w.lower() for w in words]
        best = None
        for tokens in ref_tokens:
            sm = difflib.SequenceMatcher(None, bare, [w.lower() for w, _ in tokens], autojunk=False)
            coverage = sum(s for _, _, s in sm.get_matching_blocks()) / len(bare)
            if best is None or coverage > best[0]:
                best = (coverage, tokens, sm)
        coverage, tokens, sm = best
        if coverage < 0.6:
            out.append((box, " ".join(words)))
            continue
        line_to_ref = {}
        for a, b, size in sm.get_matching_blocks():
            for t in range(size):
                line_to_ref[a + t] = b + t
        rebuilt = [w + (tokens[line_to_ref[j]][1] if j in line_to_ref else "")
                   for j, w in enumerate(words)]
        out.append((box, _space_dashes(" ".join(rebuilt))))
    return out


def build_split(split_dir: Path, base: Path, references=None, limit: int = 0):
    xml_dir = next(split_dir.glob("**/xml"), None)
    pages_dir = next(split_dir.glob("**/pages"), None)
    if xml_dir is None or pages_dir is None:
        print(f"  {split_dir.name}: missing xml={xml_dir} or pages={pages_dir}")
        return []
    page_index = {p.name: p for p in pages_dir.rglob("*") if p.suffix.lower() in IMG_EXT}
    crops_dir = base / "crops" / split_dir.name
    crops_dir.mkdir(parents=True, exist_ok=True)

    records, skipped, k = [], 0, 0
    for xml_path in sorted(xml_dir.glob("*.xml")):
        try:
            root_el = _parse_xml(xml_path)
        except Exception as e:
            if skipped == 0:
                print(f"  skip {xml_path.name}: {e}")
            skipped += 1
            continue
        src = page_index.get(_image_filename(root_el) or "")
        if src is None:
            continue
        lines = _lines(root_el)
        placed = recover(lines, references) if references else [(b, " ".join(w)) for b, w in lines]
        page = Image.open(src).convert("RGB")
        width, height = page.size
        for (x0, y0, x1, y1), text in placed:
            x0i, y0i = max(0, int(x0)), max(0, int(y0))
            x1i, y1i = min(width, int(x1)), min(height, int(y1))
            if x1i - x0i < 4 or y1i - y0i < 4:
                continue
            crop = page.crop((x0i, y0i, x1i, y1i))
            rel = crops_dir / f"{split_dir.name}_{k:06d}.png"
            crop.save(rel)
            records.append((rel.relative_to(base).as_posix(), text))
            k += 1
            if limit and len(records) >= limit:
                return records
    print(f"  {split_dir.name}: pages={len(page_index)} crops={len(records)} skipped_xml={skipped}")
    return records


def download(root="data/cvl", url=DEFAULT_URL, preview=False, official_split=False, punct=True):
    base_root = ROOT / root
    extract(download_archive(url, base_root / "cvl-database-1-1.zip"), base_root)
    base = next(base_root.glob("**/trainset"), None)
    base = base.parent if base is not None else base_root

    references = None
    if punct:
        from cvl_texts import REFERENCES
        references = REFERENCES

    per_split = {}
    for split_dir_name, key in (("trainset", "train"), ("testset", "test")):
        split_dir = next(base_root.glob(f"**/{split_dir_name}"), None)
        if split_dir is not None:
            per_split[key] = build_split(split_dir, base, references, limit=10 if preview else 0)

    if official_split:
        outputs = per_split
    else:                                          # CVL is an EN training source -> everything to train
        outputs = {"train": [r for recs in per_split.values() for r in recs]}
        stale = base / "cvl_test.tsv"
        if stale.exists() and not preview:
            stale.unlink()

    summary = {"root": str(base)}
    for key, records in outputs.items():
        summary[key] = len(records)
        if preview:
            for path, text in records[:10]:
                print(f"  {path}\t{text}")
        else:
            (base / f"cvl_{key}.tsv").write_text(
                "".join(f"{p}\t{t}\n" for p, t in records), encoding="utf-8")
    return summary


def inspect(root="data/cvl"):
    from collections import Counter
    base_root = ROOT / root
    print("base:", base_root, "| exists:", base_root.exists())

    dir_names, exts, samples, xmls = Counter(), Counter(), [], []
    for p in base_root.rglob("*"):
        if p.is_dir():
            dir_names[p.name] += 1
        else:
            exts[p.suffix.lower()] += 1
            if p.suffix.lower() == ".xml":
                xmls.append(p)
            elif len(samples) < 12:
                samples.append(str(p.relative_to(base_root)))
    print("dir names :", dict(dir_names.most_common(20)))
    print("file exts :", dict(exts))
    print("sample files:")
    for s in samples:
        print("  ", s)

    if not xmls:
        print("no .xml found — transcripts live elsewhere (check exts above)")
        return
    xml_path = xmls[0]
    print("xml file  :", xml_path.relative_to(base_root))
    el_root = _parse_xml(xml_path)
    tags, attrs = Counter(), Counter()
    for el in el_root.iter():
        tags[_strip_ns(el.tag)] += 1
        for key in el.attrib:
            attrs[_strip_ns(key)] += 1
    print("tags  :", dict(tags))
    print("attrs :", dict(attrs))

    attrtypes, per_type, tokens, no_text = Counter(), {}, [], 0
    for el in el_root.iter():
        if _strip_ns(el.tag) == "AttrRegion":
            at = el.get("attrType")
            attrtypes[at] += 1
            txt = _text_of(el)
            per_type.setdefault(at, txt)
            if txt:
                tokens.append(txt)
            else:
                no_text += 1
    print("attrType counts        :", dict(attrtypes))
    print("sample text per attrType:", per_type)
    print("regions without text   :", no_text)
    print("text tokens (1 page)   :", tokens)


def _coverage(words, references):
    bare = [w.lower() for w in words]
    best = 0.0
    for ref in references:
        sm = difflib.SequenceMatcher(None, bare, [w.lower() for w, _ in _ref_tokens(ref)], autojunk=False)
        best = max(best, sum(s for _, _, s in sm.get_matching_blocks()) / max(1, len(bare)))
    return best


def dump_texts(root="data/cvl", scan=400):
    from collections import Counter
    from cvl_texts import REFERENCES
    base_root = ROOT / root
    xml_dir = next(base_root.glob("**/trainset/**/xml"), None) or next(base_root.glob("**/xml"), None)
    counts, seen = Counter(), {}
    for xml_path in sorted(xml_dir.glob("*.xml"))[:scan]:
        lines = _lines(_parse_xml(xml_path))
        counts[sum(len(ws) for _, ws in lines)] += 1
        seen.setdefault(sum(len(ws) for _, ws in lines), (xml_path.name, lines))
    print("page word-counts:", counts.most_common())
    for n, (name, lines) in sorted(seen.items()):
        cov = _coverage([w for _, ws in lines for w in ws], REFERENCES)
        status = f"punct OK (cov={cov:.2f})" if cov >= 0.6 else f"FALLBACK words-only (cov={cov:.2f})"
        print(f"\n{n} words [{name}] -> {status}")
        for _, text in recover(lines, REFERENCES)[:4]:
            print("   ", text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/cvl")
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--no-punct", dest="punct", action="store_false",
                    help="keep words only (skip punctuation recovery)")
    ap.add_argument("--official-split", action="store_true",
                    help="keep CVL's train/test split (default: merge all into train)")
    ap.add_argument("--inspect", action="store_true", help="print real XML tags/attrs + dirs, then exit")
    ap.add_argument("--dump-texts", action="store_true",
                    help="print distinct page word-sequences (to add references), then exit")
    args = ap.parse_args()
    if args.inspect:
        inspect(args.root)
        return
    if args.dump_texts:
        dump_texts(args.root)
        return
    print("cvl:", download(args.root, args.url, args.preview, args.official_split, args.punct))


if __name__ == "__main__":
    main()
