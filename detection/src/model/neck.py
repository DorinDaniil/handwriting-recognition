"""FPN + Adaptive Scale Fusion (ASF) from DBNet++ (arXiv:2202.10304).

FPN produces P2..P5 at stride 4. ASF then re-weights each level via
stage+spatial attention before concatenation — giving scale robustness.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    """Standard top-down FPN: all outputs upsampled to stride 4 (C2 size)."""

    def __init__(self, in_channels: Sequence[int], inner_channels: int = 256,
                 out_channels: int = 256):
        super().__init__()
        self.reduce = nn.ModuleList([
            nn.Conv2d(c, inner_channels, kernel_size=1, bias=False)
            for c in in_channels
        ])
        self.smooth = nn.ModuleList([
            nn.Conv2d(inner_channels, out_channels // 4, kernel_size=3, padding=1, bias=False)
            for _ in in_channels
        ])
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, feats: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        # feats: c2, c3, c4, c5 (ascending stride)
        c2, c3, c4, c5 = feats
        p5 = self.reduce[3](c5)
        p4 = self.reduce[2](c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.reduce[1](c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p2 = self.reduce[0](c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")

        outs = [
            self.smooth[0](p2),
            self.smooth[1](p3),
            self.smooth[2](p4),
            self.smooth[3](p5),
        ]
        # bring everything to p2 size
        target_size = outs[0].shape[-2:]
        outs = [
            outs[0],
            F.interpolate(outs[1], size=target_size, mode="nearest"),
            F.interpolate(outs[2], size=target_size, mode="nearest"),
            F.interpolate(outs[3], size=target_size, mode="nearest"),
        ]
        return outs


class SpatialAttention(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_ch, 1, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.conv2(F.relu(self.conv1(x), inplace=True)))


class ASF(nn.Module):
    """Adaptive Scale Fusion. Stage attention over 4 levels + spatial attention."""

    def __init__(self, in_ch: int, num_levels: int = 4):
        super().__init__()
        self.num_levels = num_levels
        # conv over concat features → per-level stage weights
        self.stage_conv = nn.Sequential(
            nn.Conv2d(in_ch * num_levels, in_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )
        self.spatial = SpatialAttention(in_ch)
        self.weight_conv = nn.Conv2d(in_ch, num_levels, 1, bias=False)

    def forward(self, feats: Sequence[torch.Tensor]) -> torch.Tensor:
        # feats: list of N tensors, same spatial size, same channels
        cat = torch.cat(list(feats), dim=1)                # (B, N*C, H, W)
        fused = self.stage_conv(cat)                       # (B, C, H, W)
        attn = self.spatial(fused) * fused                 # spatial gating
        weights = torch.sigmoid(self.weight_conv(attn))    # (B, N, H, W)
        return torch.cat(
            [f * weights[:, i:i + 1] for i, f in enumerate(feats)],
            dim=1,
        )
