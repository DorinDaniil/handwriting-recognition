"""Page model: photo [N, 3, S, S] -> logits [N, 2, S/4, S/4] — `paper` and `core` (see pages.py).

MobileNetV3-small pretrained on ImageNet as the encoder (there are only a couple of thousand
annotated photos), a top-down decoder that brings its features back to 1/4 resolution.
About 1M parameters.
"""
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small

STAGES = ((0, 2, 16), (2, 4, 24), (4, 9, 48), (9, 12, 96))     # slices of the encoder and their channels: /4 /8 /16 /32


def block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))


class PageNet(nn.Module):
    def __init__(self, width: int = 64, pretrained: bool = True):
        super().__init__()
        encoder = mobilenet_v3_small(weights="IMAGENET1K_V1" if pretrained else None).features
        self.stages = nn.ModuleList(encoder[a:b] for a, b, _ in STAGES)
        self.lateral = nn.ModuleList(nn.Conv2d(c, width, 1) for _, _, c in STAGES)
        self.merge = nn.ModuleList(block(width, width) for _ in STAGES[:-1])
        self.head = nn.Sequential(block(width, width), nn.Conv2d(width, 2, 1))

    def forward(self, x):
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        y = self.lateral[-1](features[-1])
        for k in range(len(features) - 2, -1, -1):
            y = F.interpolate(y, size=features[k].shape[-2:], mode="bilinear", align_corners=False)
            y = self.merge[k](self.lateral[k](features[k]) + y)
        return self.head(y)
