import torch
from torch import nn

from .basenet import BaseBackbone, ConvNetArgs, get_convnet


class ResNetBackbone(BaseBackbone):
    """ResNet backbone returning pooled features and intermediates outputs.

    - Contains a single convnet, initialized by ``args``,
    - ``forward_layerwise(x)`` returns keys {"l1", "l2", "l3", "l4", "features"}.

    Args:
        args (ConvNetArgs): Arguments specifying the convnet architecture and options.
    """

    def __init__(self, convnet_args: ConvNetArgs):
        super().__init__(convnet_args)
        net, feat_dim = get_convnet(convnet_args)

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.convnet_out_dim = feat_dim

    def forward_layerwise(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        features = self.pool(x4).flatten(1)
        return {"l1": x1, "l2": x2, "l3": x3, "l4": x4, "features": features}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.forward_layerwise(x)["features"]
        return feats

    @property
    def out_dim(self) -> int:
        return self.convnet_out_dim

    @property
    def feature_dim(self) -> int:
        return self.convnet_out_dim

# class BranchResNetBackbone(BaseBackbone):
#    xxx