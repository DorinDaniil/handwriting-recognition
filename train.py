"""Unified training entrypoint.

    python train.py                               # default config.yaml
    python train.py config.yaml                   # custom config path
    python train.py trainer.epochs=200 data.batch_size=4   # CLI overrides

On first run, if split files don't exist, they are generated deterministically
from `data.labels_txt` using `data.val_fraction` and `experiment.seed`.
"""
from __future__ import annotations

import logging
import random
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.augmentation import AugConfig
from src.dataset import PaddleOCRDetDataset, detection_collate, parse_labels_txt
from src.loss import DBLoss, LossConfig
from src.model import build_model
from src.target_gen import TargetConfig
from src.trainer import Trainer
from src.utils import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train")


def _parse_args(argv: list[str]) -> tuple[Path, list[str]]:
    """First positional (if exists and ends in .yaml/.yml) = config path;
    the rest are dotlist CLI overrides."""
    cfg_path = Path("config.yaml")
    overrides: list[str] = []
    for a in argv:
        if a.endswith((".yaml", ".yml")) and "=" not in a:
            cfg_path = Path(a)
        else:
            overrides.append(a)
    return cfg_path, overrides


def _build_aug_cfg(cfg) -> AugConfig:
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
    )


def _build_target_cfg(cfg) -> TargetConfig:
    t = cfg.target
    return TargetConfig(shrink_ratio=t.shrink_ratio,
                        thresh_min=t.thresh_min,
                        thresh_max=t.thresh_max)


def _build_loss_cfg(cfg) -> LossConfig:
    l = cfg.loss
    return LossConfig(alpha=l.alpha, beta=l.beta, ohem_ratio=l.ohem_ratio,
                      bce_weight=l.bce_weight, dice_weight=l.dice_weight,
                      score_weighting=l.score_weighting)


def _ensure_splits(cfg) -> None:
    train_path = Path(cfg.data.train_split)
    val_path = Path(cfg.data.val_split)
    if train_path.exists() and val_path.exists():
        return
    logger.info("generating train/val splits")
    items = parse_labels_txt(Path(cfg.data.labels_txt))
    paths = [rel for rel, _ in items]
    rng = random.Random(cfg.experiment.seed)
    rng.shuffle(paths)
    n_val = max(1, int(len(paths) * cfg.data.val_fraction))
    val = set(paths[:n_val])
    train_path.parent.mkdir(parents=True, exist_ok=True)
    val_path.parent.mkdir(parents=True, exist_ok=True)
    with open(train_path, "w", encoding="utf-8") as f:
        f.writelines(p + "\n" for p in paths if p not in val)
    with open(val_path, "w", encoding="utf-8") as f:
        f.writelines(p + "\n" for p in paths if p in val)
    logger.info(f"train={len(paths) - n_val}  val={n_val}")


def main() -> None:
    cfg_path, overrides = _parse_args(sys.argv[1:])
    logger.info(f"loading config: {cfg_path}")
    cfg = OmegaConf.load(cfg_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    logger.info("\n" + OmegaConf.to_yaml(cfg))

    set_seed(cfg.experiment.seed)

    _ensure_splits(cfg)

    # datasets
    train_aug = _build_aug_cfg(cfg)
    val_aug = AugConfig(**{**train_aug.__dict__, "tier": "none"})
    target_cfg = _build_target_cfg(cfg)

    train_set = PaddleOCRDetDataset(
        dataset_root=cfg.data.dataset_root,
        labels_txt=cfg.data.labels_txt,
        aug_cfg=train_aug,
        target_cfg=target_cfg,
        split_file=cfg.data.train_split,
        min_score=cfg.data.min_score,
        train=True,
    )
    val_set = PaddleOCRDetDataset(
        dataset_root=cfg.data.dataset_root,
        labels_txt=cfg.data.labels_txt,
        aug_cfg=val_aug,
        target_cfg=target_cfg,
        split_file=cfg.data.val_split,
        min_score=cfg.data.min_score,
        train=False,
    )
    logger.info(f"dataset: train={len(train_set)} val={len(val_set)}")

    train_loader = DataLoader(
        train_set, batch_size=cfg.data.batch_size, shuffle=True,
        num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory,
        collate_fn=detection_collate, drop_last=True, persistent_workers=cfg.data.num_workers > 0,
    )
    val_loader = DataLoader(
        val_set, batch_size=max(1, cfg.data.batch_size // 2), shuffle=False,
        num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory,
        collate_fn=detection_collate, drop_last=False, persistent_workers=cfg.data.num_workers > 0,
    )

    # model + loss
    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    if device.type != cfg.device:
        logger.warning(f"requested device={cfg.device} but CUDA unavailable -> using {device}")
    model = build_model(cfg)
    loss_fn = DBLoss(_build_loss_cfg(cfg))

    # trainer
    out_dir = Path(cfg.experiment.output_dir) / cfg.experiment.name
    trainer = Trainer(
        model=model, loss_fn=loss_fn,
        train_loader=train_loader, val_loader=val_loader,
        cfg=cfg, device=device, output_dir=out_dir,
    )
    # persist resolved config
    OmegaConf.save(cfg, out_dir / "config.resolved.yaml")
    trainer.fit()


if __name__ == "__main__":
    main()
