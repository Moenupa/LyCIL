from abc import abstractmethod
from dataclasses import dataclass
from typing import Literal

import torch
import torchvision.models as tvm
from torch import nn


@dataclass
class ConvNetArgs:
    """Configuration for a single convolutional network.

    Attributes:
        name (Literal["resnet18", "resnet34", "resnet50"]):
            ResNet variant to instantiate. (default: ``"resnet18"``)
        pretrained (bool):
            If ``True``, load ImageNet-pretrained weights. (default: ``False``)
        cifar (bool):
            If ``True``, apply CIFAR-specific modifications. (default: ``False``)
    """

    name: Literal["resnet18", "resnet34", "resnet50"] = "resnet18"
    pretrained: bool = False
    cifar: bool = False


def get_convnet(args: ConvNetArgs) -> tuple[tvm.resnet.ResNet, int]:
    """Initialize a convnet according to given ``args``.

    Args:
        args (ConvNetArgs): Arguments specifying the convnet architecture and options.

    Returns:
        tuple[tvm.resnet.ResNet, int]: A 2-tuple of:
            - Instantiated convnet module (e.g., ResNet).
            - Feature dimension (number of output channels of the last feature layer,
            equal to ``in_features`` of the classifier head).

    Raises:
        ValueError: If ``args.name`` is not a recognized ResNet variant.
    """
    # match-case introduced in python 3.10, we specified >=3.10 in pyproject.toml
    # double-check the env if you have issues with this.
    match args.name:
        case "resnet18":
            net = tvm.resnet18(
                weights=tvm.ResNet18_Weights.IMAGENET1K_V1 if args.pretrained else None
            )
            feat_dim = 512
        case "resnet34":
            net = tvm.resnet34(
                weights=tvm.ResNet34_Weights.IMAGENET1K_V1 if args.pretrained else None
            )
            feat_dim = 512
        case "resnet50":
            net = tvm.resnet50(
                weights=tvm.ResNet50_Weights.IMAGENET1K_V2 if args.pretrained else None
            )
            feat_dim = 2048
            if args.cifar:
                net.conv1 = nn.Conv2d(
                    3, 64, kernel_size=3, stride=1, padding=2, bias=False
                )
                net.maxpool = nn.Identity()
                net.fc = nn.Identity()
            # net.load_state_dict(state, strict=False)
        case _:
            raise ValueError(f"Unknown ResNet variant: {args.name}")

    return net, feat_dim


class BaseBackbone(nn.Module):
    """Abstract backbone class, without classifier heads.

    All backbones should inherit from this and implement:

    - ``forward_layerwise(x)``: Callable[[Tensor], dict[str, Tensor]],
    - ``forward(x)``: Callable[[Tensor], Tensor],
    - ``out_dim``: int, output feature dimension of each convnet,
    - ``feature_dim``: int, dimension of self.forward() output.

    Args:
        convnet_args (ConvNetArgs): Args for initializing convnets.
    """

    def __init__(self, convnet_args: ConvNetArgs):
        super().__init__()
        self.convnet_args = convnet_args

    @abstractmethod
    def forward_layerwise(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run forward pass and collect intermediate-layer outputs as a dict.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            dict[str, torch.Tensor]:
                Outputs of each layer. Must have key "features" for final feature vectors.
                Example: ``{"l1": f1, "l2": f2, ..., "features": features}``.
        """
        ...

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

    @property
    @abstractmethod
    def out_dim(self) -> int:
        """Feature dimension per convnet, equal to convnet's ``out_features``."""
        ...

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Feature dimension in total, equal to classifier head's ``in_features``."""
        ...
