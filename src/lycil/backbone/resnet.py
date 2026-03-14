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


class BranchResNetBackbone(ResNetBackbone):
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
        self.convnet_out_dim = feat_dim

    def forward_layerwise(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)
        # layer-wised feature
        x1 = self.net.layer1(x)
        x2 = self.net.layer2(x1)
        x3 = self.net.layer3(x2)
        x4 = self.net.layer4(x3)
        features = self.net.avgpool(x4).flatten(1)
        return {"l1": x1, "l2": x2, "l3": x3, "l4": x4, "features": features}


    def prepare_branches(self, task_id: int | None) -> None:
        """Prepare branch params for current task.

        - task 0: disable branches, train main path.
        - task > 0: reset branches, enable parallel branches, train branch only.
        """
        if self._branches_prepared:
            return

        for p in self.net.parameters():
            p.requires_grad = True

        if task_id is not None and task_id > 0:
            self.net.reset_branches_params()

            for name, p in self.net.named_parameters():
                if "parallel_branch" not in name:
                    p.requires_grad = False

            self.net.set_branches_mode("parallel")
        else:
            self.net.set_branches_mode(None)

        self._branches_prepared = True

    @torch.no_grad()
    def compress_branches(self) -> None:
        """Merge parallel branch params into the main conv weights.

        After compression:
        - main_branch absorbs branch params
        - branch params are reset to zero
        - branch execution is disabled
        """
        for module in self.net.modules():
            if not hasattr(module, "parallel_branch"):
                continue
            if not hasattr(module, "main_branch"):
                continue

            branch = getattr(module, "parallel_branch", None)
            main = getattr(module, "main_branch", None)
            if branch is None or main is None:
                continue

            if main.weight.shape == branch.weight.shape:
                main.weight.data.add_(branch.weight.data)
            else:
                raise RuntimeError(
                    f"Cannot compress branch: weight shape mismatch: "
                    f"main={tuple(main.weight.shape)}, "
                    f"branch={tuple(branch.weight.shape)}"
                )

            if main.bias is not None and branch.bias is not None:
                main.bias.data.add_(branch.bias.data)
            elif main.bias is None and branch.bias is not None:
                raise RuntimeError("Cannot compress branch bias into bias-free main conv.")

            branch.weight.data.zero_()
            if branch.bias is not None:
                branch.bias.data.zero_()

        self.net.set_branches_mode(None)
        self._branches_prepared = False