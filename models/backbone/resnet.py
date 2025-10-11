from typing import Tuple
import torch
import torch.nn as nn
import torchvision.models as tvm

class ResNetBackbone(nn.Module):
    """ResNet backbone returning pooled features and classifier-ready dim."""
    def __init__(self, name: str = "resnet18", pretrained: bool = False):
        super().__init__()
        assert name in ["resnet18", "resnet34", "resnet50"], "Unsupported backbone"
        if name == "resnet18":
            net = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            feat_dim = 512
        elif name == "resnet34":
            net = tvm.resnet34(weights=tvm.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
            feat_dim = 512
        else:
            net = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            feat_dim = 2048
        # keep everything up to global pooling
        self.features = nn.Sequential(
            net.conv1, net.bn1, net.relu, net.maxpool,
            net.layer1, net.layer2, net.layer3, net.layer4
        )
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.out_dim = feat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return x

    @property
    def feature_dim(self) -> int:
        return self.out_dim
