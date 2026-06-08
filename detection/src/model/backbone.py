"""Backbone factory for DBNet++.

Three encoders are supported, all returning four feature maps C2..C5 at
strides 4 / 8 / 16 / 32:

    - resnet18     : 12M backbone params. Fast, good baseline.
                     Supports DCNv2 in stages 3-5 (the original DBNet++ trick).
    - resnet50     : 25M backbone params. Stronger than r18 on harder pages.
                     Same DCNv2 hook (replaces the 3x3 in each Bottleneck).
    - convnext_tiny: 28M backbone params, ~SOTA modern CNN. Tends to beat
                     ResNet-50 at the same FLOPs on dense prediction.
                     No DCN (ConvNeXt uses 7x7 depthwise convs; DCN doesn't
                     map cleanly onto them). `use_dcn` is ignored.

Pick via `model.backbone.name` in config.yaml.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torchvision
from torchvision.ops import DeformConv2d


# --- DCNv2 building blocks ------------------------------------------------

class DCNv2Block(nn.Module):
    """A drop-in replacement for a plain 3x3 conv: learns offsets + modulation."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        # 2*kh*kw offsets + kh*kw modulation masks  (3x3 -> 27 channels)
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
    """ResNet BasicBlock (used by ResNet-18) with DCNv2 as the second 3x3."""
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


class DCNBottleneck(nn.Module):
    """ResNet Bottleneck (used by ResNet-50) with DCNv2 as the middle 3x3."""
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1,
                 downsample: nn.Module | None = None):
        super().__init__()
        width = planes
        self.conv1 = nn.Conv2d(inplanes, width, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = DCNv2Block(width, width, stride=stride)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


# --- helpers to swap DCN in pretrained resnet layers ----------------------

def _replace_basicblock_with_dcn(layer: nn.Sequential) -> nn.Sequential:
    from torchvision.models.resnet import BasicBlock
    new_blocks: list[nn.Module] = []
    for b in layer:
        assert isinstance(b, BasicBlock), f"Expected BasicBlock, got {type(b)}"
        inplanes = b.conv1.in_channels
        planes = b.conv1.out_channels
        stride = b.conv1.stride[0]
        nb = DCNBasicBlock(inplanes, planes, stride=stride, downsample=b.downsample)
        nb.conv1.load_state_dict(b.conv1.state_dict())
        nb.bn1.load_state_dict(b.bn1.state_dict())
        nb.bn2.load_state_dict(b.bn2.state_dict())
        # warm-start: copy original 3x3 weights into the DCN's internal conv
        # (offsets stay 0, so DCN initially behaves like the pretrained conv)
        nb.conv2.dcn.weight.data.copy_(b.conv2.weight.data)
        new_blocks.append(nb)
    return nn.Sequential(*new_blocks)


def _replace_bottleneck_with_dcn(layer: nn.Sequential) -> nn.Sequential:
    from torchvision.models.resnet import Bottleneck
    new_blocks: list[nn.Module] = []
    for b in layer:
        assert isinstance(b, Bottleneck), f"Expected Bottleneck, got {type(b)}"
        inplanes = b.conv1.in_channels
        planes = b.conv1.out_channels
        stride = b.conv2.stride[0]
        nb = DCNBottleneck(inplanes, planes, stride=stride, downsample=b.downsample)
        nb.conv1.load_state_dict(b.conv1.state_dict())
        nb.bn1.load_state_dict(b.bn1.state_dict())
        nb.bn2.load_state_dict(b.bn2.state_dict())
        nb.conv3.load_state_dict(b.conv3.state_dict())
        nb.bn3.load_state_dict(b.bn3.state_dict())
        # warm-start the DCN with the pretrained 3x3 — same kernel, offsets=0
        nb.conv2.dcn.weight.data.copy_(b.conv2.weight.data)
        new_blocks.append(nb)
    return nn.Sequential(*new_blocks)


# --- backbones ------------------------------------------------------------

class _ResNetBackbone(nn.Module):
    """Shared wrapper around torchvision ResNets returning C2..C5."""

    out_channels: tuple[int, int, int, int]

    def __init__(self, net: nn.Module, use_dcn: bool, dcn_stages: Sequence[bool],
                 dcn_replacer):
        super().__init__()
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1  # stride 4   (stage 2)
        self.layer2 = net.layer2  # stride 8   (stage 3)
        self.layer3 = net.layer3  # stride 16  (stage 4)
        self.layer4 = net.layer4  # stride 32  (stage 5)

        if use_dcn:
            # dcn_stages indexes stages 1..5; stage 1 is stem (skip)
            if len(dcn_stages) >= 3 and dcn_stages[2]:
                self.layer2 = dcn_replacer(self.layer2)
            if len(dcn_stages) >= 4 and dcn_stages[3]:
                self.layer3 = dcn_replacer(self.layer3)
            if len(dcn_stages) >= 5 and dcn_stages[4]:
                self.layer4 = dcn_replacer(self.layer4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c2, c3, c4, c5


class ResNet18Backbone(_ResNetBackbone):
    out_channels = (64, 128, 256, 512)

    def __init__(self, pretrained: bool = True, use_dcn: bool = True,
                 dcn_stages: Sequence[bool] = (False, False, True, True, True)):
        weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        net = torchvision.models.resnet18(weights=weights)
        super().__init__(net, use_dcn=use_dcn, dcn_stages=dcn_stages,
                         dcn_replacer=_replace_basicblock_with_dcn)


class ResNet34Backbone(_ResNetBackbone):
    """ResNet-34 — same BasicBlock structure as R18, deeper (~21M params).
    Often a sweet spot between R18 speed and R50 capacity."""
    out_channels = (64, 128, 256, 512)

    def __init__(self, pretrained: bool = True, use_dcn: bool = True,
                 dcn_stages: Sequence[bool] = (False, False, True, True, True)):
        weights = torchvision.models.ResNet34_Weights.DEFAULT if pretrained else None
        net = torchvision.models.resnet34(weights=weights)
        super().__init__(net, use_dcn=use_dcn, dcn_stages=dcn_stages,
                         dcn_replacer=_replace_basicblock_with_dcn)


class ResNet50Backbone(_ResNetBackbone):
    out_channels = (256, 512, 1024, 2048)

    def __init__(self, pretrained: bool = True, use_dcn: bool = True,
                 dcn_stages: Sequence[bool] = (False, False, True, True, True)):
        weights = torchvision.models.ResNet50_Weights.DEFAULT if pretrained else None
        net = torchvision.models.resnet50(weights=weights)
        super().__init__(net, use_dcn=use_dcn, dcn_stages=dcn_stages,
                         dcn_replacer=_replace_bottleneck_with_dcn)


class ConvNeXtTinyBackbone(nn.Module):
    """ConvNeXt-Tiny feature extractor returning C2..C5 at strides 4/8/16/32.

    No DCN — ConvNeXt uses 7x7 depthwise convs and DCN doesn't fit. The
    `use_dcn` / `dcn_stages` flags are accepted but ignored.
    """

    out_channels = (96, 192, 384, 768)

    def __init__(self, pretrained: bool = True, **_ignored):
        super().__init__()
        weights = torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        net = torchvision.models.convnext_tiny(weights=weights)
        # torchvision convnext.features is a Sequential of 8 modules:
        #   0: stem (Conv 4x4 stride 4 + LayerNorm)
        #   1: stage1   (96 ch,  stride 4)
        #   2: downsample
        #   3: stage2   (192 ch, stride 8)
        #   4: downsample
        #   5: stage3   (384 ch, stride 16)
        #   6: downsample
        #   7: stage4   (768 ch, stride 32)
        f = net.features
        self.stem = f[0]
        self.stage1 = f[1]
        self.down1 = f[2]
        self.stage2 = f[3]
        self.down2 = f[4]
        self.stage3 = f[5]
        self.down3 = f[6]
        self.stage4 = f[7]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x = self.stem(x)
        c2 = self.stage1(x)            # stride 4
        x = self.down1(c2)
        c3 = self.stage2(x)            # stride 8
        x = self.down2(c3)
        c4 = self.stage3(x)            # stride 16
        x = self.down3(c4)
        c5 = self.stage4(x)            # stride 32
        return c2, c3, c4, c5


# --- factory --------------------------------------------------------------

_BACKBONES = {
    "resnet18": ResNet18Backbone,
    "resnet34": ResNet34Backbone,
    "resnet50": ResNet50Backbone,
    "convnext_tiny": ConvNeXtTinyBackbone,
}


def build_backbone(bb_cfg) -> nn.Module:
    """Instantiate a backbone from a config node (`cfg.model.backbone`)."""
    name = str(bb_cfg.name).lower()
    if name not in _BACKBONES:
        raise ValueError(
            f"Unknown backbone: {bb_cfg.name!r}. "
            f"Available: {sorted(_BACKBONES.keys())}"
        )
    cls = _BACKBONES[name]
    return cls(
        pretrained=bool(getattr(bb_cfg, "pretrained", True)),
        use_dcn=bool(getattr(bb_cfg, "use_dcn", False)),
        dcn_stages=tuple(getattr(bb_cfg, "dcn_stages", (False, False, True, True, True))),
    )
