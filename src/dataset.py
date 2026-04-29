"""PaddleOCR labels.txt dataset for DBNet++ training.

Each line in labels.txt looks like:

    <rel_path>\t<JSON list of {"transcription", "points", "score"} items>

This dataset:
    - reads labels.txt into an index,
    - optionally filters by score (min_score),
    - applies Augmenter to image + polygons,
    - generates DBNet++ targets via target_gen.generate_targets,
    - also propagates per-polygon scores so the loss can down-weight weak boxes.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .augmentation import Augmenter, AugConfig, SyntheticPaperAugmenter
from .target_gen import TargetConfig, _shrink_polygon, generate_targets

logger = logging.getLogger(__name__)


def expand_polygon_height(poly: np.ndarray, fraction: float, image_shape: tuple[int, int]) -> np.ndarray:
    """Expand a text polygon along its short rotated-rect axis."""
    if fraction <= 0 or len(poly) < 3:
        return poly
    rect = cv2.minAreaRect(poly.astype(np.float32))
    (_, _), (w, h), angle = rect
    if w < 1 or h < 1:
        return poly
    theta = np.deg2rad(angle)
    width_axis = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
    height_axis = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float32)
    normal = height_axis if h <= w else width_axis
    center = poly.mean(axis=0)
    signs = np.sign((poly - center) @ normal).astype(np.float32)
    signs[signs == 0] = 1.0
    delta = min(w, h) * float(fraction)
    expanded = poly.astype(np.float32) + signs[:, None] * normal[None, :] * delta
    image_h, image_w = image_shape
    expanded[:, 0] = np.clip(expanded[:, 0], 0, image_w - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, image_h - 1)
    return expanded.astype(np.float32)


def parse_labels_txt(labels_txt: Path) -> list[tuple[str, list[dict]]]:
    """Parse PaddleOCR labels.txt. Returns list of (rel_path, boxes)."""
    items: list[tuple[str, list[dict]]] = []
    with open(labels_txt, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            rel, payload = line.split("\t", 1)
            items.append((rel, json.loads(payload)))
    return items


def read_split(split_path: Path | None) -> set[str] | None:
    """Read newline-separated relative paths. Returns None if split_path is None."""
    if split_path is None:
        return None
    with open(split_path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


class PaddleOCRDetDataset(Dataset):
    """DBNet++ detection dataset from PaddleOCR labels.txt."""

    def __init__(
        self,
        dataset_root: str | Path,
        labels_txt: str | Path,
        aug_cfg: AugConfig,
        target_cfg: TargetConfig,
        split_file: str | Path | None = None,
        min_score: float = 0.0,
        train: bool = True,
        synthetic_paper_aug: bool = False,
        table_backgrounds_dir: str | Path | None = None,
        skip_main_train_geometry: bool = False,
    ):
        self.root = Path(dataset_root)
        self.min_score = min_score
        self.train = train
        self.target_cfg = target_cfg
        self.aug_cfg = aug_cfg
        self.augmenter = Augmenter(aug_cfg, train=(train and not skip_main_train_geometry))
        self.synthetic_paper_augmenter = (
            SyntheticPaperAugmenter(aug_cfg.image_size, table_backgrounds_dir)
            if train and synthetic_paper_aug
            else None
        )

        all_items = parse_labels_txt(Path(labels_txt))
        allowed = read_split(Path(split_file) if split_file else None)
        if allowed is not None:
            self.items = [it for it in all_items if it[0] in allowed]
        else:
            self.items = all_items

    # --------------------------------------------------------------- helpers

    def _load_image(self, rel: str) -> np.ndarray:
        abs_path = self.root / rel
        if not abs_path.exists():
            raise FileNotFoundError(abs_path)
        img = cv2.imread(str(abs_path), cv2.IMREAD_COLOR)
        if img is None:
            # fall back to PIL for odd extensions / unicode paths
            from PIL import Image
            img = np.array(Image.open(abs_path).convert("RGB"))
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _filter_boxes(self, boxes: list[dict]) -> tuple[list[np.ndarray], list[float]]:
        polys: list[np.ndarray] = []
        scores: list[float] = []
        for b in boxes:
            s = b.get("score") or 1.0
            if s < self.min_score:
                continue
            pts = np.asarray(b["points"], dtype=np.float32).reshape(-1, 2)
            if len(pts) < 3:
                continue
            polys.append(pts)
            scores.append(float(s))
        return polys, scores

    # --------------------------------------------------------------- API

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        last_error: Exception | None = None
        for offset in range(len(self.items)):
            safe_idx = (idx + offset) % len(self.items)
            rel, boxes = self.items[safe_idx]
            try:
                return self._getitem_loaded(rel, boxes)
            except (FileNotFoundError, OSError, ValueError) as exc:
                last_error = exc
                if offset < 5:
                    logger.warning("skip unreadable image %s: %s", self.root / rel, exc)
                continue
        raise RuntimeError("No readable images left in dataset") from last_error

    def _getitem_loaded(self, rel: str, boxes: list[dict]) -> dict:
        image = self._load_image(rel)
        polys, scores = self._filter_boxes(boxes)

        if self.synthetic_paper_augmenter is not None:
            image, polys, pre_keep = self.synthetic_paper_augmenter(image, polys)
            scores = [s for s, k in zip(scores, pre_keep) if k]

        image_t, polys_t, keep = self.augmenter(image, polys)
        scores_t = [s for s, k in zip(scores, keep) if k]
        if self.aug_cfg.box_expand_height_fraction > 0:
            polys_t = [
                expand_polygon_height(poly, self.aug_cfg.box_expand_height_fraction, image_t.shape[:2])
                for poly in polys_t
            ]

        h, w = image_t.shape[:2]
        targets = generate_targets(
            image_shape=(h, w),
            polygons=polys_t,
            ignore_flags=[False] * len(polys_t),
            cfg=self.target_cfg,
        )

        # per-polygon score map: for each shrunk positive pixel, remember its
        # source polygon score. Useful for score-weighted loss.
        score_map = np.ones((h, w), dtype=np.float32)
        if polys_t:
            tmp = np.zeros((h, w), dtype=np.float32)
            for poly, s in zip(polys_t, scores_t):
                shrunk = _shrink_polygon(poly, self.target_cfg.shrink_ratio)
                if shrunk is None:
                    continue
                cv2.fillPoly(tmp, [shrunk.astype(np.int32)], float(s))
            # where tmp>0 use its value, else keep 1.0 (so background BCE weight = 1)
            score_map = np.where(tmp > 0, tmp, 1.0).astype(np.float32)

        # HWC -> CHW, float32
        image_chw = np.transpose(image_t, (2, 0, 1)).astype(np.float32)

        return {
            "image": torch.from_numpy(image_chw),
            "prob_map": torch.from_numpy(targets["prob_map"]),
            "prob_mask": torch.from_numpy(targets["prob_mask"]),
            "thresh_map": torch.from_numpy(targets["thresh_map"]),
            "thresh_mask": torch.from_numpy(targets["thresh_mask"]),
            "score_map": torch.from_numpy(score_map),
            "polys": polys_t,           # python list, used only for debugging/eval
            "scores": scores_t,
            "rel_path": rel,
        }


def detection_collate(batch: Sequence[dict]) -> dict:
    """Stack tensor fields, keep list fields as Python lists."""
    tensor_keys = ("image", "prob_map", "prob_mask", "thresh_map", "thresh_mask", "score_map")
    out: dict = {k: torch.stack([b[k] for b in batch], dim=0) for k in tensor_keys}
    out["polys"] = [b["polys"] for b in batch]
    out["scores"] = [b["scores"] for b in batch]
    out["rel_path"] = [b["rel_path"] for b in batch]
    return out
