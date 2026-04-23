"""DBNet++ loss.

L = alpha * L_s (prob) + beta * L_t (thresh) + L_b (binary)

L_s:  BCE with OHEM (neg:pos ratio = ohem_ratio) + Dice — on probability map.
L_t:  masked L1 — only on the ring between shrunk and expanded polygons.
L_b:  Dice on differentiable binary map = sigmoid(k * (P - T)).

Extensions:
- score_weighting: per-pixel weight from score_map (noisy pseudo-label -> lower weight)
- prob_mask:       binary mask of pixels that count toward L_s / L_b (ignore regions).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossConfig:
    alpha: float = 1.0
    beta: float = 10.0
    ohem_ratio: float = 3.0
    bce_weight: float = 1.0
    dice_weight: float = 1.0
    score_weighting: bool = True


def ohem_bce(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
             neg_pos_ratio: float = 3.0,
             per_pixel_weight: torch.Tensor | None = None) -> torch.Tensor:
    """
    BCE with online hard example mining.
    pred, target, mask: (B, 1, H, W). per_pixel_weight: same shape or None.
    """
    pred = pred.clamp(min=1e-6, max=1 - 1e-6)
    bce = F.binary_cross_entropy(pred, target, reduction="none")
    if per_pixel_weight is not None:
        bce = bce * per_pixel_weight

    pos_mask = (target > 0.5) * mask
    neg_mask = (target <= 0.5) * mask

    pos_loss = bce * pos_mask
    neg_loss_all = bce * neg_mask

    num_pos = int(pos_mask.sum().item())
    num_neg_all = int(neg_mask.sum().item())
    num_neg = min(num_neg_all, int(max(num_pos, 1) * neg_pos_ratio))
    if num_neg <= 0:
        neg_loss = neg_loss_all.sum() * 0.0  # zero tensor
    else:
        neg_loss_flat = neg_loss_all.view(-1)
        topk, _ = torch.topk(neg_loss_flat, k=num_neg)
        neg_loss = topk.sum()

    denom = max(num_pos + num_neg, 1)
    return (pos_loss.sum() + neg_loss) / denom


def dice_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
              per_pixel_weight: torch.Tensor | None = None,
              eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice on the masked region."""
    if per_pixel_weight is None:
        per_pixel_weight = torch.ones_like(pred)
    w = mask * per_pixel_weight
    intersection = (pred * target * w).sum()
    denom = (pred * w).sum() + (target * w).sum() + eps
    return 1.0 - (2.0 * intersection + eps) / denom


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = (pred - target).abs() * mask
    denom = mask.sum().clamp(min=1.0)
    return diff.sum() / denom


class DBLoss(nn.Module):
    def __init__(self, cfg: LossConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict:
        cfg = self.cfg
        prob = preds["prob"]
        thresh = preds["thresh"]
        binary = preds.get("binary")

        # (B, H, W) -> (B, 1, H, W)
        prob_map = batch["prob_map"].unsqueeze(1)
        prob_mask = batch["prob_mask"].unsqueeze(1)
        thresh_map = batch["thresh_map"].unsqueeze(1)
        thresh_mask = batch["thresh_mask"].unsqueeze(1)
        score_map = batch["score_map"].unsqueeze(1) if cfg.score_weighting else None

        l_bce = ohem_bce(prob, prob_map, prob_mask,
                         neg_pos_ratio=cfg.ohem_ratio,
                         per_pixel_weight=score_map)
        l_dice = dice_loss(prob, prob_map, prob_mask, per_pixel_weight=score_map)
        l_s = cfg.bce_weight * l_bce + cfg.dice_weight * l_dice

        l_t = masked_l1(thresh, thresh_map, thresh_mask)

        if binary is not None:
            l_b = dice_loss(binary, prob_map, prob_mask, per_pixel_weight=score_map)
        else:
            l_b = torch.zeros((), device=prob.device)

        total = cfg.alpha * l_s + cfg.beta * l_t + l_b
        return {
            "loss": total,
            "l_s": l_s.detach(),
            "l_t": l_t.detach(),
            "l_b": l_b.detach(),
            "l_bce": l_bce.detach(),
            "l_dice": l_dice.detach(),
        }
