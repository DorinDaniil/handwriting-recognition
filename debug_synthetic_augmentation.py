from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.augmentation import AugConfig, IMAGENET_MEAN, IMAGENET_STD
from src.dataset import PaddleOCRDetDataset
from src.target_gen import TargetConfig


def build_aug_cfg(cfg) -> AugConfig:
    a = cfg.aug
    return AugConfig(
        tier=a.tier,
        image_size=cfg.data.image_size,
        p_rot90=a.p_rot90,
        p_hflip=a.p_hflip,
        p_vflip=a.p_vflip,
        p_color_jitter=a.p_color_jitter,
        p_blur=a.p_blur,
        p_noise=a.p_noise,
        box_jitter_px=a.box_jitter_px,
        box_expand_height_fraction=a.get("box_expand_height_fraction", 0.0),
    )


def build_target_cfg(cfg) -> TargetConfig:
    t = cfg.target
    return TargetConfig(
        shrink_ratio=t.shrink_ratio,
        thresh_min=t.thresh_min,
        thresh_max=t.thresh_max,
    )


def tensor_image_to_uint8(image_chw) -> np.ndarray:
    image = image_chw.numpy().transpose(1, 2, 0)
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    image = (image * std + mean) * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def draw_polygons(image: np.ndarray, polygons: list[np.ndarray]) -> np.ndarray:
    output = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for idx, poly in enumerate(polygons):
        pts = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
        color = ((37 * idx) % 255, (131 * idx) % 255, (211 * idx) % 255)
        cv2.polylines(output, [pts], isClosed=True, color=color, thickness=2)
        x, y = pts.reshape(-1, 2)[0]
        cv2.putText(output, str(idx), (int(x), max(0, int(y) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug synthetic paper augmentation used during DBNet training.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "debug_synthetic_augmentation")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--start-index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    synthetic_cfg = cfg.get("synthetic", {})
    if not bool(synthetic_cfg.get("enabled", False)):
        raise RuntimeError("synthetic.enabled=false in config")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = PaddleOCRDetDataset(
        dataset_root=synthetic_cfg.dataset_root,
        labels_txt=synthetic_cfg.labels_txt,
        aug_cfg=build_aug_cfg(cfg),
        target_cfg=build_target_cfg(cfg),
        split_file=None,
        min_score=cfg.data.min_score,
        train=True,
        synthetic_paper_aug=bool(synthetic_cfg.get("paper_aug", True)),
        table_backgrounds_dir=synthetic_cfg.get("table_backgrounds_dir", None),
        skip_main_train_geometry=True,
    )

    for i in range(args.count):
        idx = (args.start_index + i) % len(dataset)
        sample = dataset[idx]
        image = tensor_image_to_uint8(sample["image"])
        debug = draw_polygons(image, sample["polys"])
        output_path = args.output_dir / f"synthetic_aug_{idx:06d}.jpg"
        cv2.imwrite(str(output_path), debug)
        print(f"[OK] {output_path} boxes={len(sample['polys'])} rel={sample['rel_path']}")


if __name__ == "__main__":
    main()
