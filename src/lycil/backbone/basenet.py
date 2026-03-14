from abc import abstractmethod
from dataclasses import dataclass
from typing import Literal

import torch
import torchvision.models as tvm
from torch import nn

from . import branchnet as bnm


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



def _apply_cifar_mods(net: nn.Module) -> None:
    """Convert ImageNet-style ResNet stem into a CIFAR-style stem."""
    if hasattr(net, 'conv1'):
        net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    if hasattr(net, 'maxpool'):
        net.maxpool = nn.Identity()
    if hasattr(net, 'fc'):
        net.fc = nn.Identity()



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
        case _:
            raise ValueError(f"Unknown ResNet variant: {args.name}")

    if args.cifar:
        _apply_cifar_mods(net)

    return net, feat_dim



def get_branch_convnet(args: ConvNetArgs) -> tuple[bnm.ResNet, int]:
    """Initialize a branch-enabled ResNet.

    Notes:
        - Branch modules are created in ``parallel`` mode so the auxiliary branch
          weights exist in the model graph.
        - The caller can later toggle them on/off via ``set_branches_mode``.
    """
    match args.name:
        case "resnet18":
            net = bnm.resnet18(
                weights=bnm.ResNet18_Weights.IMAGENET1K_V1 if args.pretrained else None,
                branch_mode=None,
            )
            feat_dim = 512
        case "resnet34":
            net = bnm.resnet34(
                weights=bnm.ResNet34_Weights.IMAGENET1K_V1 if args.pretrained else None,
                branch_mode=None,
            )
            feat_dim = 512
        case "resnet50":
            net = bnm.resnet50(
                weights=bnm.ResNet50_Weights.IMAGENET1K_V2 if args.pretrained else None,
                branch_mode=None,
            )
            feat_dim = 2048
        case _:
            raise ValueError(f"Unknown ResNet variant: {args.name}")

    if args.cifar:
        _apply_cifar_mods(net)

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
