import torch
from torch import nn

from .basenet import BaseBackbone, ConvNetArgs, get_convnet


class DERNetBackbone(BaseBackbone):
    def __init__(self, convnet_args: ConvNetArgs):
        super().__init__(convnet_args)
        self.convnets = nn.ModuleList()

    def forward_layerwise(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = torch.cat([conv(x) for conv in self.convnets], 1)
        return {"features": features}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.forward_layerwise(x)["features"]
        return feats

    @property
    def out_dim(self) -> int:
        if len(self.convnets) == 0:
            raise RuntimeError(
                "DERNetBackbone not initialized yet, call ``prepare_for_new_task()`` first."
            )

        return self.convnets[-1].out_dim  # ty: ignore[invalid-return-type]

    @property
    def feature_dim(self) -> int:
        return self.out_dim * len(self.convnets)

    @torch.no_grad()
    def prepare_for_new_task(self):
        new_convnet, _ = get_convnet(self.convnet_args)

        if len(self.convnets) > 0:
            # init new from last convnet
            last_convnet = self.convnets[-1]
            new_convnet.load_state_dict(last_convnet.state_dict())

        self.convnets.append(new_convnet)
