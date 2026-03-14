import torch
from torch import nn

from .basenet import BaseBackbone, ConvNetArgs, get_convnet, get_branch_convnet


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


class BranchResNetBackbone(BaseBackbone):
    """Branch-enabled ResNet backbone.

    Main points:
    - The backbone owns a single branch-capable ResNet.
    - The extra branch weights are created during initialization.
    - ``prepare_for_new_task()`` re-initializes branch params and enables the
      parallel branch path.
    - ``freeze_main_path()`` can be used to train only branch parameters.
    """

    def __init__(self, convnet_args: ConvNetArgs):
        super().__init__(convnet_args)
        net, feat_dim = get_branch_convnet(convnet_args)
        self.net = net

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.convnet_out_dim = feat_dim

        # Keep branch modules instantiated but disabled by default.
        self.disable_branches()

    def forward_layerwise(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        features = self.pool(x4).flatten(1)
        return {"l1": x1, "l2": x2, "l3": x3, "l4": x4, "features": features}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_layerwise(x)["features"]

    def enable_branches(self) -> None:
        self.net.set_branches_mode("parallel")

    def disable_branches(self) -> None:
        self.net.set_branches_mode(None)

    @torch.no_grad()
    def prepare_for_new_task(self) -> None:
        """Re-initialize branch parameters and turn branch path on."""
        self.net.reset_branches_params()
        self.enable_branches()

    def freeze_main_path(self) -> None:
        """Freeze original parameters and leave branch params trainable."""
        for name, param in self.net.named_parameters():
            param.requires_grad = "parallel_branch" in name

    def unfreeze_all(self) -> None:
        for param in self.net.parameters():
            param.requires_grad = True

    @property
    def out_dim(self) -> int:
        return self.convnet_out_dim

    @property
    def feature_dim(self) -> int:
        return self.convnet_out_dim