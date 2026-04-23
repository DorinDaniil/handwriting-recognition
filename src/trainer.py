"""Self-contained DBNet++ trainer: AMP, cosine+warmup LR, EMA, checkpointing."""
from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .loss import DBLoss
from .postprocess import PostprocessConfig, decode_prob_map
from .utils import ModelEMA, hmean_metric

logger = logging.getLogger(__name__)


def _amp_dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def _cosine_lr(step: int, total_steps: int, warmup_steps: int,
               base_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def build_optimizer(model: nn.Module, cfg: Any) -> torch.optim.Optimizer:
    tcfg = cfg.trainer
    params = [p for p in model.parameters() if p.requires_grad]
    if tcfg.optimizer.lower() == "adamw":
        return torch.optim.AdamW(params, lr=tcfg.lr,
                                 betas=tuple(tcfg.betas),
                                 weight_decay=tcfg.weight_decay)
    if tcfg.optimizer.lower() == "sgd":
        return torch.optim.SGD(params, lr=tcfg.lr, momentum=0.9,
                               weight_decay=tcfg.weight_decay)
    raise ValueError(f"Unknown optimizer: {tcfg.optimizer}")


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        loss_fn: DBLoss,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: Any,
        device: torch.device,
        output_dir: Path,
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)

        self.optimizer = build_optimizer(self.model, cfg)
        self.use_amp = bool(cfg.trainer.amp)
        self.amp_dtype = _amp_dtype(cfg.trainer.amp_dtype)
        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.use_amp and self.amp_dtype == torch.float16))

        self.ema = ModelEMA(self.model, decay=cfg.trainer.ema_decay) if cfg.trainer.ema else None

        self.total_steps = cfg.trainer.epochs * len(train_loader)
        self.warmup_steps = cfg.trainer.warmup_epochs * len(train_loader)
        self.global_step = 0
        self.start_epoch = 0
        self.best_hmean = 0.0

        self.post_cfg = PostprocessConfig(
            thresh=cfg.postprocess.thresh,
            box_thresh=cfg.postprocess.box_thresh,
            unclip_ratio=cfg.postprocess.unclip_ratio,
            max_candidates=cfg.postprocess.max_candidates,
            min_size=cfg.postprocess.min_size,
        )

        if cfg.trainer.resume:
            self._load_checkpoint(cfg.trainer.resume)

    # ------------------------------------------------------------------ core

    def _set_lr(self) -> float:
        lr = _cosine_lr(
            self.global_step, self.total_steps, self.warmup_steps,
            base_lr=self.cfg.trainer.lr, min_lr=self.cfg.trainer.min_lr,
        )
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        return lr

    def _forward_loss(self, batch: dict) -> dict:
        with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
            preds = self.model(batch["image"])
            return self.loss_fn(preds, batch)

    def _train_one_epoch(self, epoch: int) -> dict:
        self.model.train()
        start = time.time()
        meters: dict[str, float] = {}
        for step, batch in enumerate(self.train_loader):
            batch = {k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            lr = self._set_lr()

            out = self._forward_loss(batch)
            loss = out["loss"]

            self.optimizer.zero_grad(set_to_none=True)
            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.trainer.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.trainer.grad_clip)
                self.optimizer.step()

            if self.ema is not None:
                self.ema.update(self.model)

            # rolling meters
            for k, v in out.items():
                meters[k] = meters.get(k, 0.0) + float(v.detach() if torch.is_tensor(v) else v)
            self.global_step += 1

            if (step + 1) % self.cfg.trainer.log_every == 0:
                denom = step + 1
                msg = " | ".join(f"{k}={meters[k] / denom:.4f}" for k in meters)
                logger.info(f"epoch {epoch} step {step + 1}/{len(self.train_loader)} "
                            f"lr={lr:.2e} | {msg}")

        denom = max(len(self.train_loader), 1)
        avg = {k: v / denom for k, v in meters.items()}
        avg["epoch_time_s"] = time.time() - start
        return avg

    # ------------------------------------------------------------------ eval

    @torch.no_grad()
    def evaluate(self) -> dict:
        model = self.ema.module if self.ema is not None else self.model
        model.eval()
        pred_polys: list[list] = []
        gt_polys: list[list] = []
        for batch in self.val_loader:
            images = batch["image"].to(self.device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                out = model(images)
            prob = out["prob"].float().squeeze(1).cpu().numpy()
            H, W = images.shape[-2:]
            for i in range(images.shape[0]):
                preds, _ = decode_prob_map(prob[i], (H, W), self.post_cfg)
                pred_polys.append(preds)
                gts = batch["polys"][i]  # list of np.ndarray already in net-resolution
                gt_polys.append([g for g in gts])
        return hmean_metric(pred_polys, gt_polys, iou_thresh=0.5)

    # ------------------------------------------------------------------ checkpoints

    def _save_checkpoint(self, name: str, extra: dict | None = None) -> None:
        path = self.output_dir / "checkpoints" / name
        state = {
            "model": self.model.state_dict(),
            "ema": self.ema.module.state_dict() if self.ema is not None else None,
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "global_step": self.global_step,
            "best_hmean": self.best_hmean,
        }
        if extra:
            state.update(extra)
        torch.save(state, path)
        logger.info(f"saved checkpoint: {path}")

    def _load_checkpoint(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        if self.ema is not None and ckpt.get("ema") is not None:
            self.ema.module.load_state_dict(ckpt["ema"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.global_step = ckpt["global_step"]
        self.best_hmean = ckpt.get("best_hmean", 0.0)
        self.start_epoch = ckpt.get("epoch", 0) + 1
        logger.info(f"resumed from {path} @ epoch {self.start_epoch}")

    # ------------------------------------------------------------------ loop

    def fit(self) -> None:
        history: list[dict] = []
        for epoch in range(self.start_epoch, self.cfg.trainer.epochs):
            train_stats = self._train_one_epoch(epoch)
            log = {"epoch": epoch, **train_stats}

            if (epoch + 1) % self.cfg.trainer.eval_every == 0:
                val = self.evaluate()
                log["val"] = val
                logger.info(f"[eval] epoch {epoch} precision={val['precision']:.4f} "
                            f"recall={val['recall']:.4f} hmean={val['hmean']:.4f}")
                if val["hmean"] > self.best_hmean:
                    self.best_hmean = val["hmean"]
                    self._save_checkpoint("best.pt", extra={"epoch": epoch, "val": val})

            if (epoch + 1) % self.cfg.trainer.save_every == 0:
                self._save_checkpoint("last.pt", extra={"epoch": epoch})

            history.append(log)
            with open(self.output_dir / "history.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(log, default=float) + "\n")

        self._save_checkpoint("last.pt", extra={"epoch": self.cfg.trainer.epochs - 1})
