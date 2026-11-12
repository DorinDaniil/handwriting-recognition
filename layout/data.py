"""Training samples for the page model.

Photos: ai-forever School Notebooks (COCO annotations, category `paper`: one polygon per page;
many photos are spreads). Scans (optional): folders of single-page images that fill the frame —
exam sheets; a sample puts one of them on a table, or two side by side as a spread, so the model
also learns that a full-frame scan is one page.

x: the photo in a 512 x 512 frame, normalized, [3, 512, 512]
y: [2, 128, 128] — `paper` (any page) and `core` (each page with its border peeled off)
ids: [128, 128] — which page a pixel belongs to, for the metric
"""
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from scipy import ndimage
from torch.utils.data import Dataset

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
IMAGES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def letterbox(img, size):
    """PIL image -> float32 [3, size, size] (image in the top-left corner) and its scale inside"""
    scale = size / max(img.size)
    w, h = round(img.width * scale), round(img.height * scale)
    canvas = np.zeros((size, size, 3), np.float32)
    canvas[:h, :w] = np.asarray(img.convert("RGB").resize((w, h), Image.BILINEAR), np.float32) / 255.0
    return ((canvas - MEAN) / STD).transpose(2, 0, 1), scale


def make_target(ids, shrink):
    """page ids [h, w] -> float32 [2, h, w]: paper, core. The core of a page is what is deeper
    than shrink * sqrt(page area) from its outline; the frame of the picture is not an outline."""
    paper = (ids > 0).astype(np.float32)
    core = np.zeros_like(paper)
    for k in range(1, int(ids.max()) + 1):
        page = ids == k
        if page.all():
            core[:] = 1.0
        elif page.any():
            core[ndimage.distance_transform_edt(page) > shrink * np.sqrt(page.sum())] = 1.0
    return np.stack([paper, core])


def snapshot_of(repo):
    """A HuggingFace dataset id (taken from the local cache, downloaded once) or a local folder."""
    if Path(repo).is_dir():
        return Path(repo)
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo, repo_type="dataset"))


class Notebooks:
    """Photos and page polygons of one School Notebooks snapshot for the given splits."""

    def __init__(self, repo, splits):
        snapshot = snapshot_of(repo)
        self.zip_path = sorted(snapshot.rglob("*.zip"))[0]
        self.zip = None
        with zipfile.ZipFile(self.zip_path) as zf:
            names = {Path(n).name: n for n in zf.namelist() if Path(n).suffix.lower() in IMAGES}

        self.items = []                                             # (name in the zip, [polygons of each page])
        for path in sorted(snapshot.rglob("*.json")):
            if not any(s in path.stem.lower() for s in splits):
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            paper = {c["id"] for c in data["categories"] if c["name"] == "paper"}
            pages = {}
            for ann in data["annotations"]:
                seg = ann.get("segmentation") or []
                if ann["category_id"] in paper and seg:
                    polys = seg if isinstance(seg[0], (list, tuple)) else [seg]
                    pages.setdefault(str(ann["image_id"]), []).append(
                        [list(zip(p[0::2], p[1::2])) for p in polys if len(p) >= 6])
            for image in data["images"]:
                name = names.get(Path(image["file_name"]).name)
                if name and str(image["id"]) in pages:
                    self.items.append((name, pages[str(image["id"])]))

    def __len__(self):
        return len(self.items)

    def load(self, i, side):
        """-> (photo no larger than needed for `side`, page ids [H, W] at the photo's size)"""
        if self.zip is None:                                         # one handle per dataloader worker
            self.zip = zipfile.ZipFile(self.zip_path)
        name, pages = self.items[i]
        img = Image.open(io.BytesIO(self.zip.read(name)))
        full = img.width
        img.draft("RGB", (side, side))                               # jpeg: decode straight to a smaller size
        img = img.convert("RGB")
        k = img.width / full
        ids = Image.new("L", img.size, 0)
        draw = ImageDraw.Draw(ids)
        for page, polys in enumerate(pages, 1):
            for poly in polys:
                draw.polygon([(x * k, y * k) for x, y in poly], fill=page)
        return img, ids


class Scans:
    """Single-page images that fill the frame, shown as a sheet or a spread on a table."""

    def __init__(self, dirs, p_spread, p_full):
        self.paths = sorted(p for d in dirs for p in Path(d).rglob("*") if p.suffix.lower() in IMAGES)
        self.p_spread, self.p_full = p_spread, p_full

    def __len__(self):
        return len(self.paths)

    def sheet(self, rng, side):
        img = Image.open(self.paths[int(rng.integers(len(self.paths)))])
        img.draft("RGB", (side, side))
        img = img.convert("RGB")
        img.thumbnail((side, side))
        return img

    def load(self, rng, side):
        left = self.sheet(rng, side)
        sheets = [left]
        if rng.random() < self.p_spread:
            right = self.sheet(rng, side)
            sheets.append(right.resize((max(1, round(right.width * left.height / right.height)), left.height)))
        width, height = sum(s.width for s in sheets), left.height
        pad = 0 if len(sheets) == 1 and rng.random() < self.p_full else int(rng.uniform(0.03, 0.3) * height)
        table = tuple(int(v) for v in np.clip(rng.uniform(25, 200) + rng.normal(0, 25, 3), 0, 255))
        img = Image.new("RGB", (width + 2 * pad, height + 2 * pad), table)
        ids = Image.new("L", img.size, 0)
        x = pad
        for page, s in enumerate(sheets, 1):
            img.paste(s, (x, pad))
            ids.paste(page, (x, pad, x + s.width, pad + height))
            x += s.width
        if len(sheets) == 2:                                         # shadow of the binding
            arr = np.asarray(img, np.float32)
            xs = np.arange(img.width)
            dark = rng.uniform(0.1, 0.5) * np.exp(-((xs - pad - left.width) / (0.02 * width)) ** 2)
            img = Image.fromarray((arr * (1 - dark)[None, :, None]).clip(0, 255).astype(np.uint8))
        return img, ids


def frame(img, ids, size, rng, aug):
    """Photo and its page ids -> both in a size x size frame. Without `aug` this is the plain
    letterbox of inference; with it the photo is turned, moved, scaled and seen at an angle."""
    side = max(img.size)
    quad = np.array([[0, 0], [0, side], [side, side], [side, 0]], np.float64)   # frame corners in the photo: UL, LL, LR, UR
    fill = (0, 0, 0)
    if aug is not None and rng.random() < aug.p_geometry:
        centre = np.array(img.size) / 2
        turn = np.radians(rng.uniform(-aug.rotate, aug.rotate) + 90 * int(rng.integers(4)) * (rng.random() < aug.p_rot90))
        zoom = rng.uniform(*aug.zoom)
        c, s = np.cos(turn), np.sin(turn)
        quad = (quad - side / 2) / zoom @ np.array([[c, s], [-s, c]]) + centre
        quad += rng.uniform(-aug.shift, aug.shift, 2) * side + rng.uniform(-aug.perspective, aug.perspective, (4, 2)) * side
        border = np.concatenate([np.asarray(img)[0], np.asarray(img)[-1]])
        fill = tuple(int(v) for v in np.clip(np.median(border, axis=0) + rng.normal(0, 12, 3), 0, 255))
    corners = tuple(quad.ravel())
    img = img.transform((size, size), Image.QUAD, corners, resample=Image.BILINEAR, fillcolor=fill)
    ids = ids.transform((size, size), Image.QUAD, corners, resample=Image.NEAREST, fillcolor=0)
    if aug is not None and rng.random() < 0.5:
        img, ids = img.transpose(Image.FLIP_LEFT_RIGHT), ids.transpose(Image.FLIP_LEFT_RIGHT)
    return img, ids


def colour(img, rng, aug):
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(1 - aug.brightness, 1 + aug.brightness))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(1 - aug.contrast, 1 + aug.contrast))
    img = ImageEnhance.Color(img).enhance(rng.uniform(1 - aug.saturation, 1 + aug.saturation))
    if rng.random() < aug.p_blur:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.5)))
    if rng.random() < aug.p_jpeg:
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=int(rng.integers(40, 90)))
        img = Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")
    return img


class Pages(Dataset):
    """
    train=True: notebook photos of the train/val splits plus scans (cfg.data.scan_share of the
    samples), augmented. train=False: notebook photos of the test split, framed as at inference.
    """

    def __init__(self, cfg, train):
        d = cfg.data
        splits = ("train", "val") if train else ("test",)
        self.books = [Notebooks(repo, splits) for repo in d.notebooks]
        self.index = [(b, i) for b, book in enumerate(self.books) for i in range(len(book))]
        self.scans = Scans(d.scans, d.p_spread, d.p_full) if train and d.get("scans") else None
        if self.scans is not None and len(self.scans):
            self.index += [(None, 0)] * int(len(self.index) * d.scan_share / (1 - d.scan_share))
        self.cfg, self.aug = d, cfg.aug if train else None
        counts = np.bincount([len(pages) for book in self.books for _, pages in book.items], minlength=1)
        print(f"{'train' if train else 'test'}: {len(self.index)} samples; "
              f"photos by number of pages: { {n: int(c) for n, c in enumerate(counts) if c} }")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, n):
        d, rng = self.cfg, np.random.default_rng(None if self.aug is not None else n)
        book, i = self.index[n]
        side = 2 * d.image_size
        img, ids = self.scans.load(rng, side) if book is None else self.books[book].load(i, side)
        img, ids = frame(img, ids, d.image_size, rng, self.aug)
        if self.aug is not None:
            img = colour(img, rng, self.aug)
        small = np.asarray(ids.resize((d.image_size // d.stride,) * 2, Image.NEAREST))
        x = ((np.asarray(img, np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)
        return torch.from_numpy(x), torch.from_numpy(make_target(small, d.shrink)), torch.from_numpy(small.astype(np.int64))
