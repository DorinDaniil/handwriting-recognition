"""DBNet++ model assembly."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import ResNet18Backbone
from .head import DBHead
from .neck import ASF, FPN


class DBNetPP(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        neck_in_channels: tuple[int, ...],
        inner_channels: int = 256,
        out_channels: int = 256,
        use_asf: bool = True,
        head_inner: int = 64,
        k: float = 50.0,
    ):
        super().__init__()
        self.backbone = backbone
        self.fpn = FPN(neck_in_channels, inner_channels=inner_channels,
                       out_channels=out_channels)

        per_level_ch = out_channels // 4
        if use_asf:
            self.asf: nn.Module | None = ASF(per_level_ch, num_levels=4)
        else:
            self.asf = None
        self.head = DBHead(out_channels, inner_channels=head_inner, k=k)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.backbone(image)
        fpn_outs = self.fpn(feats)              # list of 4 tensors, each (B, C/4, H/4, W/4)
        if self.asf is not None:
            x = self.asf(fpn_outs)              # (B, C, H/4, W/4)
        else:
            x = torch.cat(fpn_outs, dim=1)
        out = self.head(x)
        # head upsamples 4x -> stride 1, so prob/thresh match input resolution
        # (ensure exact match in case of odd input sizes)
        out = {k: F.interpolate(v, size=image.shape[-2:], mode="bilinear", align_corners=False)
               for k, v in out.items()}
        return out


def build_model(cfg: Any) -> DBNetPP:
    """Build DBNetPP from config (OmegaConf node with the schema of config.yaml)."""
    bb_cfg = cfg.model.backbone
    if bb_cfg.name != "resnet18":
        raise NotImplementedError(f"Backbone {bb_cfg.name} not implemented yet")
    backbone = ResNet18Backbone(
        pretrained=bool(bb_cfg.pretrained),
        use_dcn=bool(bb_cfg.use_dcn),
        dcn_stages=tuple(bb_cfg.dcn_stages),
    )
    return DBNetPP(
        backbone=backbone,
        neck_in_channels=backbone.out_channels,
        inner_channels=cfg.model.neck.inner_channels,
        out_channels=cfg.model.neck.out_channels,
        use_asf=bool(cfg.model.neck.use_asf),
        head_inner=64,
        k=float(cfg.model.head.k),
    )
