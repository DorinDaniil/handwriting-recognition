"""DB head: two twin branches produce probability map P and threshold map T.

Differentiable binarization:  B = sigmoid( k * (P - T) )

During training all three (P, T, B) are supervised.
At inference only P (and optionally T) are needed.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _upsample_branch(in_ch: int, inner: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, inner, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(inner),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(inner, inner, kernel_size=2, stride=2),
        nn.BatchNorm2d(inner),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(inner, 1, kernel_size=2, stride=2),
    )


class DBHead(nn.Module):
    """Twin deconv branches producing stride-1 prob + thresh maps."""

    def __init__(self, in_channels: int, inner_channels: int = 64, k: float = 50.0):
        super().__init__()
        self.k = k
        self.prob_branch = _upsample_branch(in_channels, inner_channels)
        self.thresh_branch = _upsample_branch(in_channels, inner_channels)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def db(self, p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.k * (p - t))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        p_logit = self.prob_branch(x)
        t_logit = self.thresh_branch(x)
        prob = torch.sigmoid(p_logit)
        thresh = torch.sigmoid(t_logit)
        if self.training:
            binary = self.db(prob, thresh)
            return {"prob": prob, "thresh": thresh, "binary": binary}
        return {"prob": prob, "thresh": thresh}
