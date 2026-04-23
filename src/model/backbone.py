"""ResNet-18 backbone with optional DCNv2 (from torchvision.ops) in later stages.

Returns C2, C3, C4, C5 feature maps at strides 4, 8, 16, 32.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torchvision
from torchvision.ops import DeformConv2d


class DCNv2Block(nn.Module):
    """A drop-in replacement for a plain 3x3 conv: learns offsets + modulation."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        # 2*kh*kw offsets + kh*kw modulation masks
        self.offset = nn.Conv2d(in_ch, 3 * 3 * 3, kernel_size=3,
                                stride=stride, padding=1)
        self.dcn = DeformConv2d(in_ch, out_ch, kernel_size=3,
                                stride=stride, padding=1, bias=False)
        # init offsets to zero for stable training
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = self.offset(x)
        offset, mask = o[:, :18], o[:, 18:].sigmoid()
        return self.dcn(x, offset, mask)


class DCNBasicBlock(nn.Module):
    """ResNet BasicBlock where the second 3x3 is DCNv2."""
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1,
                 downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = DCNv2Block(planes, planes, stride=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


def _replace_basicblock_with_dcn(layer: nn.Sequential) -> nn.Sequential:
    """Clone a torchvision resnet layer and swap BasicBlocks for DCNBasicBlocks."""
    from torchvision.models.resnet import BasicBlock
    new_blocks: list[nn.Module] = []
    for b in layer:
        assert isinstance(b, BasicBlock), f"Expected BasicBlock, got {type(b)}"
        inplanes = b.conv1.in_channels
        planes = b.conv1.out_channels
        stride = b.conv1.stride[0]
        # downsample inherited as-is (stride only in first block)
        nb = DCNBasicBlock(inplanes, planes, stride=stride, downsample=b.downsample)
        # transfer first conv weights for a warm start
        nb.conv1.load_state_dict(b.conv1.state_dict())
        nb.bn1.load_state_dict(b.bn1.state_dict())
        nb.bn2.load_state_dict(b.bn2.state_dict())
        new_blocks.append(nb)
    return nn.Sequential(*new_blocks)


class ResNet18Backbone(nn.Module):
    """Torchvision ResNet-18 feature extractor returning C2..C5."""

    out_channels = (64, 128, 256, 512)

    def __init__(self, pretrained: bool = True, use_dcn: bool = True,
                 dcn_stages: Sequence[bool] = (False, False, True, True, True)):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        net = torchvision.models.resnet18(weights=weights)

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1  # stride 4,  64 ch   (stage 2)
        self.layer2 = net.layer2  # stride 8, 128 ch   (stage 3)
        self.layer3 = net.layer3  # stride 16, 256 ch  (stage 4)
        self.layer4 = net.layer4  # stride 32, 512 ch  (stage 5)

        if use_dcn:
            # dcn_stages indexes stages 1..5; stage 1 is stem (skip)
            if dcn_stages[2]:
                self.layer2 = _replace_basicblock_with_dcn(self.layer2)
            if dcn_stages[3]:
                self.layer3 = _replace_basicblock_with_dcn(self.layer3)
            if dcn_stages[4]:
                self.layer4 = _replace_basicblock_with_dcn(self.layer4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c2, c3, c4, c5
