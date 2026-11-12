"""Data sources, built from the `data.sources` block of a config.

Every source has a name, a language tag, a sampling weight, `draw(rng, step) -> (image, text)`
for training and, when the data has a held-out part, a `test` list of (image, text).

  tsv    crops + manifest "path<TAB>text"            (Cyrillic, School Notebooks, CVL, IMGUR5K)
  pairs  a folder of png and a folder of txt          (HWR200)
  hf     a HuggingFace line dataset                   (IAM)
  pages  page images + labels.txt with line polygons  (essays); lines are cropped on demand
  synth  the font-based generator from recognition/src/synth
  blank  crops with nothing to read: empty paper or show-through of the other side
"""
from __future__ import annotations

import json
import random
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from .augment import Photometric, add_context
from .geometry import Frame, crop_line, order_quad, paper_colour
from .text import Alphabet


class Shuffled:
    """Endless passes over range(n), each pass in a new random order."""

    def __init__(self, n: int):
        self.n, self.order, self.at = n, None, 0

    def next(self, rng) -> int:
        if self.order is None or self.at == self.n:
            self.order, self.at = rng.permutation(self.n), 0
        self.at += 1
        return int(self.order[self.at - 1])


class Files:
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        path, text = self.pairs[i]
        return Image.open(path).convert("RGB"), text


class Rows:
    """Rows of a HuggingFace split; `kept` maps our index to the row, `texts` are already cleaned."""

    def __init__(self, split, image_key, kept, texts):
        self.split, self.image_key, self.kept, self.texts = split, image_key, kept, texts

    def __len__(self):
        return len(self.kept)

    def __getitem__(self, i):
        return self.split[self.kept[i]][self.image_key].convert("RGB"), self.texts[i]


class Lines:
    """Pre-cropped line images."""

    def __init__(self, name, spec, train, test, augment):
        self.name, self.lang, self.weight = name, spec.get("lang", "ru"), float(spec.get("weight", 1.0))
        self.train, self.test = train, test
        self.context = augment.get("context") if spec.get("context", True) else None
        self.photometric = Photometric(augment.get("photometric", 0.0))
        self.order = Shuffled(len(train))

    def start(self, worker: int):
        self.order = Shuffled(len(self.train))

    def draw(self, rng, step):
        image, text = self.train[self.order.next(rng)]
        c = self.context
        if c is not None and rng.random() < c.p:
            donor = self.train[int(rng.integers(len(self.train)))][0] if rng.random() < c.neighbour else None
            image = add_context(image, rng, c.x, c.y, donor)
        return self.photometric(image, rng), text


@lru_cache(maxsize=32)
def load_page(path: str):
    page = Image.open(path).convert("RGB")
    return page, paper_colour(page)


class PageLines:
    """Lines of labelled pages. A record is (page path, labelled quad, text, detector quad or None).

    When the detector found the line, the crop is made from the detector's own box — exactly what
    the recogniser gets at inference. Indexing gives the fixed inference crop; `draw` jitters it.
    """

    def __init__(self, records, frame):
        self.records, self.frame = records, frame

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        path, labelled, text, detected = self.records[i]
        page, fill = load_page(path)
        if detected is not None:
            return crop_line(page, detected, Frame.uniform(*self.frame.infer), fill), text
        x, y = (sum(side) / 2 for side in (self.frame.labelled.x, self.frame.labelled.y))
        return crop_line(page, labelled, Frame.uniform(x, y), fill), text

    def draw(self, i, rng, p_detected=0.8):
        path, labelled, text, detected = self.records[i]
        page, fill = load_page(path)
        if detected is not None and rng.random() < p_detected:
            quad, span = detected, self.frame.detector
        else:
            quad, span = labelled, self.frame.labelled
        return crop_line(page, quad, Frame.random(rng, span.x, span.y), fill), text


class Pages:
    def __init__(self, name, spec, train, test, augment):
        self.name, self.lang, self.weight = name, spec.get("lang", "ru"), float(spec.get("weight", 1.0))
        self.train, self.test = train, test
        self.photometric = Photometric(augment.get("photometric", 0.0))
        self.order = Shuffled(len(train))

    def start(self, worker: int):
        self.order = Shuffled(len(self.train))

    def draw(self, rng, step):
        image, text = self.train.draw(self.order.next(rng), rng)
        return (self.photometric(image, rng), text) if image is not None else None


class SynthVal:
    """A fixed synthetic set: the same image for the same index at every evaluation."""

    def __init__(self, synth, n: int):
        self.synth, self.n = synth, n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        gen = self.synth.gen
        return gen.sample(self.synth.make_rng(self.synth.seed + 1, 0, i), step=gen.cfg.warmup_steps)


class Synth:
    def __init__(self, name, spec, root: Path):
        recognition = str((root / spec.get("recognition_root", "../recognition")).resolve())
        if recognition not in sys.path:
            sys.path.insert(0, recognition)
        from src.synth import HandwrittenLineGenerator, make_generator

        self.name, self.lang, self.weight = name, "synth", float(spec.get("weight", 1.0))
        self.seed, self.make_rng, self.worker, self.count = int(spec.get("seed", 42)), make_generator, 0, 0
        self.gen = HandwrittenLineGenerator.from_dirs(
            ru_text_dirs=list(spec.get("ru_text_dirs", [])), en_text_dirs=list(spec.get("en_text_dirs", [])),
            ru_font_dirs=list(spec.ru_font_dirs), en_font_dirs=list(spec.en_font_dirs),
            ru_text_weights=list(spec.get("ru_text_weights", [])),
            en_text_weights=list(spec.get("en_text_weights", [])),
            p_ru=spec.get("p_ru", 0.6), len_chars=tuple(spec.get("len_chars", (4, 100))),
            p_hyphenate=spec.get("p_hyphenate", 0.15), seed=self.seed,
            curriculum=bool(spec.get("curriculum", True)), warmup_steps=int(spec.get("warmup_steps", 8000)),
            synth_overrides=spec.get("overrides"))
        self.test = SynthVal(self, int(spec.val_samples)) if spec.get("val_samples") else None

    def start(self, worker: int):
        self.worker, self.count = worker, 0
        self.gen.warm_cache()

    def draw(self, rng, step):
        self.count += 1
        return self.gen.sample(self.make_rng(self.seed, self.worker, self.count), step=step)


class Blank:
    """Teaches the model to stay silent on boxes the detector should not have produced."""
    test = None

    def __init__(self, name, spec, synth: Synth):
        self.name, self.lang, self.weight = name, "synth", float(spec.get("weight", 1.0))
        self.gen = synth.gen

    def start(self, worker: int):
        pass

    def draw(self, rng, step):
        gen, t = self.gen, self.gen.difficulty(step)
        height = int(rng.integers(48, 121))
        width = int(height * rng.uniform(1.5, 14.0))
        paper = gen.paper.make((width, height), rng, t).convert("RGBA")
        if rng.random() < 0.7:
            text, lang = gen.sampler.sample(rng, t)
            entry = gen.fonts.sample(rng, lang)
            lo, hi = gen.cfg.font.sizes_px
            ink, _ = gen.renderer.render(entry.filter(text), gen.fonts.get(entry, int(rng.integers(lo, hi + 1))), rng, t)
            scale = height * rng.uniform(0.5, 0.9) / max(ink.height, 1)
            ghost = ink.resize((max(1, int(ink.width * scale)), max(1, int(ink.height * scale))), Image.BILINEAR)
            ghost = ghost.transpose(Image.FLIP_LEFT_RIGHT).filter(ImageFilter.GaussianBlur(rng.uniform(0.6, 1.8)))
            faded = np.array(ghost)
            faded[..., 3] = (faded[..., 3] * rng.uniform(0.06, 0.3)).astype(np.uint8)
            layer = Image.new("RGBA", paper.size, (0, 0, 0, 0))
            layer.paste(Image.fromarray(faded),
                        (int(rng.integers(-ghost.width // 2, max(1, width - ghost.width // 2))),
                         int(rng.integers(0, max(1, height - ghost.height + 1)))))
            paper = Image.alpha_composite(paper, layer)
        return Image.fromarray(gen.effects(np.asarray(paper.convert("RGB")), rng, t)), ""


def _pairs_from_tsv(tsv: Path, alphabet: Alphabet) -> list:
    pairs = []
    for row in tsv.read_text(encoding="utf-8").splitlines():
        name, tab, text = row.partition("\t")
        text = alphabet.clean(text) if tab else None
        if not text:
            continue
        image = Path(name.strip())
        for candidate in (image, tsv.parent / image, tsv.parent / tsv.stem / image.name, tsv.parent / image.name):
            if candidate.is_file():
                pairs.append((str(candidate), text))
                break
    return pairs


def _tsv(name, spec, root, alphabet, augment):
    base = root / spec.root
    train = next(base.glob(spec.get("train", "train.tsv")), None)
    test = next(base.glob(spec.test), None) if spec.get("test") else None
    if train is None:
        return None
    return Lines(name, spec, Files(_pairs_from_tsv(train, alphabet)),
                 Files(_pairs_from_tsv(test, alphabet)) if test else None, augment)


def _pairs(name, spec, root, alphabet, augment):
    """Same split as recognition/src/finetune: sorted names, shuffled with the seed, head is the test."""
    images, texts = Path(spec.images), Path(spec.texts)
    pairs = []
    for image in sorted(images.glob("*.png")):
        label = texts / (image.stem + ".txt")
        if label.exists():
            text = label.read_text(encoding="utf-8").strip()
            if text:
                pairs.append((str(image), text))
    if not pairs:
        return None
    random.Random(spec.get("seed", 42)).shuffle(pairs)
    n_test = int(len(pairs) * spec.get("test_frac", 0.1))
    clean = [[(path, text) for path, text in ((p, alphabet.clean(t)) for p, t in part) if text]
             for part in (pairs[n_test:], pairs[:n_test])]
    return Lines(name, spec, Files(clean[0]), Files(clean[1]) if clean[1] else None, augment)


def _hf(name, spec, root, alphabet, augment):
    from datasets import concatenate_datasets, load_dataset

    cache = spec.get("cache_dir")
    data = load_dataset(spec.hf_id, cache_dir=str(root / cache) if cache else None)
    image_key, text_key = spec.get("image_key", "image"), spec.get("text_key", "text")
    train_name, test_name = spec.get("train_split", "train"), spec.get("test_split", "test")

    def rows(split):
        texts = [alphabet.clean(str(t)) for t in split[text_key]]
        kept = [i for i, t in enumerate(texts) if t]
        return Rows(split, image_key, kept, [texts[i] for i in kept])

    parts = [data[train_name]] + [data[s] for s in data if s not in (train_name, test_name) and "val" in s.lower()]
    train = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    return Lines(name, spec, rows(train), rows(data[test_name]) if test_name in data else None, augment)


def read_labels(labels: Path, root: Path) -> dict:
    """labels.txt (PaddleOCR format) -> {page path: [(quad, transcription), ...]} in file order."""
    pages = {}
    for row in labels.read_text(encoding="utf-8").splitlines():
        rel, tab, payload = row.partition("\t")
        page = root / rel.strip()
        if not tab or not page.exists():
            continue
        pages[str(page)] = [(order_quad(item["points"]), (item.get("transcription") or "").strip())
                            for item in json.loads(payload) if item.get("points")]
    return pages


def split_pages(pages: dict, seed: int, test_frac: float) -> set:
    """Held-out pages. Same order as recognition/finetune_essays, so a larger test_frac only
    adds pages to the old test set."""
    names = sorted(path for path, lines in pages.items() if any(text for _, text in lines))
    random.Random(seed).shuffle(names)
    return set(names[:int(len(names) * test_frac)])


def _pages(name, spec, root, alphabet, augment, frame):
    data_root = root / spec.root
    labels = data_root / spec.get("labels", "labels.txt")
    pages = read_labels(labels, data_root) if labels.exists() else {}
    if not pages:
        return None
    found = {}
    if spec.get("detections") and (data_root / spec.detections).exists():
        found = json.loads((data_root / spec.detections).read_text(encoding="utf-8"))
    held_out = split_pages(pages, spec.get("seed", 42), spec.get("test_frac", 0.1))

    train, test, matched = [], [], 0
    for path, lines in pages.items():
        boxes = found.get(Path(path).relative_to(data_root).as_posix(), [])
        for k, (quad, text) in enumerate(lines):
            text = alphabet.clean(text)
            if not text:
                continue
            detected = np.asarray(boxes[k], dtype=np.float64) if k < len(boxes) and boxes[k] else None
            matched += detected is not None
            (test if path in held_out else train).append((path, quad, text, detected))
    print(f"source '{name}': pages {len(pages)} (held out {len(held_out)}), "
          f"lines with a detector box {matched}/{len(train) + len(test)}")
    return Pages(name, spec, PageLines(train, frame), PageLines(test, frame) if test else None, augment)


def build_sources(cfg, alphabet: Alphabet, root: Path) -> list:
    root, augment, built = Path(root), cfg.augment, {}
    for name, spec in cfg.data.sources.items():
        if not spec.get("enabled", True):
            continue
        kind = spec.kind
        if kind == "synth":
            source = Synth(name, spec, root)
        elif kind == "blank":
            source = Blank(name, spec, built[spec.get("synth", "synth")])
        elif kind == "pages":
            source = _pages(name, spec, root, alphabet, augment, cfg.frame)
        elif kind in ("tsv", "pairs", "hf"):
            source = {"tsv": _tsv, "pairs": _pairs, "hf": _hf}[kind](name, spec, root, alphabet, augment)
        else:
            raise ValueError(f"source '{name}': unknown kind {kind!r}")
        if source is None:
            print(f"source '{name}': no data found, skipped")
            continue
        built[name] = source
        sizes = [len(part) if part is not None and hasattr(part, "__len__") else "-"
                 for part in (getattr(source, "train", None), source.test)]
        print(f"source '{name}' [{source.lang}] weight {source.weight}: train {sizes[0]} test {sizes[1]}")
    return list(built.values())
