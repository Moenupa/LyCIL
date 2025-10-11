from typing import Tuple, Dict
import torch
import torch.nn as nn
import torchvision.models as tvm

class ResNetBackbone(nn.Module):
    """ResNet backbone returning pooled features and intermediates for POD.

    - forward(x): returns global pooled feature (B, D)
    - forward_feats(x): returns dict of feature maps from layers { 'l2','l3','l4' }
    """
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

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.out_dim = feat_dim

    def forward_feats(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.stem(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return {"l2": x2, "l3": x3, "l4": x4}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.forward_feats(x)["l4"]
        x = self.pool(feats).flatten(1)
        return x

    @property
    def feature_dim(self) -> int:
        return self.out_dim
