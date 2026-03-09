import torch
from torch import nn

from .basenet import BaseBackbone, ConvNetArgs, get_convnet


class DERNetBackbone(BaseBackbone):
    """DER-style backbone that concatenates features from task-specific convnets.

    - Contains a bank of convnets, initialized by ``args``,
    - Convnet load from the last convnet when a new task arrives,
    - ``forward_layerwise(x)`` returns keys {"features"}.

    Args:
        args (ConvNetArgs): Arguments specifying the convnet architecture and options.
    """

    def __init__(self, convnet_args: ConvNetArgs):
        super().__init__(convnet_args)
        self.convnets = nn.ModuleList()
        self.convnet_out_dim: int | None = None

    def forward_layerwise(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = torch.cat([conv(x) for conv in self.convnets], 1)
        return {"features": features}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.forward_layerwise(x)["features"]
        return feats

    @property
    def out_dim(self) -> int:
        if len(self.convnets) == 0 or self.convnet_out_dim is None:
            raise RuntimeError(
                "DERNetBackbone not initialized yet, call ``prepare_for_new_task()`` first."
            )

        return self.convnet_out_dim

    @property
    def feature_dim(self) -> int:
        return self.out_dim * len(self.convnets)

    @torch.no_grad()
    def prepare_for_new_task(self):
        """Add a new task-specific convnet to the backbone's convnet bank.

        A fresh convnet is created from ``self.convnet_args``. If the bank is
        non-empty, the new network's weights are copied from the last convnet.
        After this call, :attr:`out_dim` and :attr:`feature_dim` are valid.
        """
        new_convnet, self.convnet_out_dim = get_convnet(self.convnet_args)

        if len(self.convnets) > 0:
            # init new from last convnet
            last_convnet = self.convnets[-1]
            new_convnet.load_state_dict(last_convnet.state_dict())

        self.convnets.append(new_convnet)
