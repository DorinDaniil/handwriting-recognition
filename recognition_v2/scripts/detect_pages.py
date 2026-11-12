#!/usr/bin/env python3
"""Run the line detector (unchanged, from detection/) over labelled pages and remember which of
its boxes is which labelled line. The `pages` source then crops training lines from the
detector's own boxes — the crops the recogniser will get at inference.

    python scripts/detect_pages.py --labels /data/kaliningrad_essays/labels.txt
    python scripts/detect_pages.py --labels .../labels.txt --det-ckpt ../detection/outputs/dbnetpp_r34_hwr/best.pt

Writes detections.json next to labels.txt: {page: [detector quad or null, one per labelled line]}.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path[:0] = [str(ROOT), str(REPO)]

import numpy as np
from PIL import Image
from tqdm import tqdm

from htr.geometry import match_detections
from htr.sources import read_labels
from infer_page import LineDetector, pick_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--det-config", type=Path, default=REPO / "detection/config.yaml")
    ap.add_argument("--det-ckpt", type=Path, default=REPO / "detection/outputs/dbnetpp_r34_hwr/best.pt")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    root = args.labels.parent
    pages = read_labels(args.labels, root)
    detector = LineDetector(args.det_config, args.det_ckpt, pick_device(args.device))

    found, lines, matched = {}, 0, 0
    for path, labelled in tqdm(pages.items(), desc="pages"):
        boxes, _ = detector.detect(np.array(Image.open(path).convert("RGB")))
        match = match_detections([quad for quad, _ in labelled], boxes)
        found[Path(path).relative_to(root).as_posix()] = [
            boxes[j].tolist() if j is not None else None for j in match]
        lines += len(match)
        matched += sum(j is not None for j in match)

    out = args.out or root / "detections.json"
    out.write_text(json.dumps(found), encoding="utf-8")
    print(f"{len(pages)} pages, {matched}/{lines} labelled lines found as one whole detector box -> {out}")


if __name__ == "__main__":
    main()
