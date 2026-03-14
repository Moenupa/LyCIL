import torch
from torch import nn
import torch.nn.functional as F
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

    def prepare_branches(self, freeze_main_branch: bool = False) -> None:
        """Prepare branch parameters for training.

        Args:
            freeze_main_branch:
                If False, disable branches and train the main path normally.
                If True, reset branch params, enable branch mode, freeze the main
                path, and train only parallel branch parameters.
        """
        for p in self.net.parameters():
            p.requires_grad = True

        if freeze_main_branch:
            self.net.reset_branches_params()
            for name, p in self.net.named_parameters():
                if "parallel_branch" not in name:
                    p.requires_grad = False

            self.net.set_branches_mode("parallel")
            self.net.eval()
        else:
            self.net.set_branches_mode(None)
            self.net.train()

    # @torch.no_grad()
    # def compress_branches(self) -> None:
    #     """Merge parallel branch params into the main branch weights.
    #
    #     Supported cases:
    #     - branch and main have the same kernel shape, e.g. 3x3 -> 3x3
    #     - branch is 1x1 and main is 3x3, branch is padded to the center
    #
    #     After compression:
    #     - main_branch absorbs branch params
    #     - parallel_branch params are reset to zero
    #     - branch execution is disabled
    #     """
    #     for module in self.net.modules():
    #         if not hasattr(module, "parallel_branch"):
    #             continue
    #         if not hasattr(module, "main_branch"):
    #             continue
    #
    #         branch = getattr(module, "parallel_branch", None)
    #         main = getattr(module, "main_branch", None)
    #         if branch is None or main is None:
    #             continue
    #
    #         if not isinstance(branch, torch.nn.Conv2d):
    #             continue
    #         if not isinstance(main, torch.nn.Conv2d):
    #             continue
    #
    #         bw = branch.weight.data
    #         mw = main.weight.data
    #
    #         if mw.shape == bw.shape:
    #             main.weight.data.add_(bw)
    #         elif bw.shape[:2] == mw.shape[:2] and bw.shape[-2:] == (1, 1) and mw.shape[-2:] == (3, 3):
    #             main.weight.data.add_(F.pad(bw, [1, 1, 1, 1], "constant", 0))
    #         else:
    #             raise RuntimeError(
    #                 "Cannot compress branch due to incompatible weight shapes: "
    #                 f"main={tuple(mw.shape)}, branch={tuple(bw.shape)}"
    #             )
    #
    #         if main.bias is not None and branch.bias is not None:
    #             main.bias.data.add_(branch.bias.data)
    #         elif main.bias is None and branch.bias is not None:
    #             raise RuntimeError(
    #                 "Cannot compress branch bias into a bias-free main branch."
    #             )
    #
    #         branch.weight.data.zero_()
    #         if branch.bias is not None:
    #             branch.bias.data.zero_()
    #
    #     self.net.set_branches_mode(None)
    def compress_branches(self) -> None:
        """Merge parallel branch params into the main branch weights.

        Supported cases:
        - branch and main have the same kernel shape, e.g. 3x3 -> 3x3
        - branch is 1x1 and main is 3x3, branch is padded to the center

        After compression:
        - main_branch absorbs branch params
        - parallel_branch params are reset to zero
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

            if not isinstance(branch, torch.nn.Conv2d):
                continue
            if not isinstance(main, torch.nn.Conv2d):
                continue

            bw = branch.weight.data
            mw = main.weight.data

            print(f"  main weight shape   : {tuple(mw.shape)}")
            print(f"  branch weight shape : {tuple(bw.shape)}")

            print("  before:")
            print(
                f"    main   sum={mw.sum().item():.6f}, mean={mw.mean().item():.6f}, absmax={mw.abs().max().item():.6f}")
            print(
                f"    branch sum={bw.sum().item():.6f}, mean={bw.mean().item():.6f}, absmax={bw.abs().max().item():.6f}")

            # 打印一个固定位置，方便看是否真的加进去了
            print(f"    sample main[0,0,...] before:\n{mw[0, 0].detach().cpu()}")
            print(f"    sample branch[0,0,...] before:\n{bw[0, 0].detach().cpu()}")

            if mw.shape == bw.shape:
                add_w = bw
                print("  merge mode: direct add")
            elif bw.shape[:2] == mw.shape[:2] and bw.shape[-2:] == (1, 1) and mw.shape[-2:] == (3, 3):
                add_w = F.pad(bw, [1, 1, 1, 1], "constant", 0)
                print("  merge mode: 1x1 -> 3x3 center pad")
                print(f"    padded branch[0,0,...]:\n{add_w[0, 0].detach().cpu()}")
            else:
                raise RuntimeError(
                    "Cannot compress branch due to incompatible weight shapes: "
                    f"main={tuple(mw.shape)}, branch={tuple(bw.shape)}"
                )

            main.weight.data.add_(add_w)

            print("  after weight merge:")
            print(
                f"    main   sum={main.weight.data.sum().item():.6f}, mean={main.weight.data.mean().item():.6f}, absmax={main.weight.data.abs().max().item():.6f}")
            print(f"    sample main[0,0,...] after:\n{main.weight.data[0, 0].detach().cpu()}")

            if main.bias is not None and branch.bias is not None:
                print("  bias merge: yes")
                print(f"    main bias before   sum={main.bias.data.sum().item():.6f}")
                print(f"    branch bias before sum={branch.bias.data.sum().item():.6f}")

                main.bias.data.add_(branch.bias.data)

                print(f"    main bias after    sum={main.bias.data.sum().item():.6f}")
            elif main.bias is None and branch.bias is not None:
                raise RuntimeError(
                    "Cannot compress branch bias into a bias-free main branch."
                )
            else:
                print("  bias merge: skipped")

            branch.weight.data.zero_()
            if branch.bias is not None:
                branch.bias.data.zero_()

            print("  after zeroing branch:")
            print(f"    branch weight sum={branch.weight.data.sum().item():.6f}")
            if branch.bias is not None:
                print(f"    branch bias sum  ={branch.bias.data.sum().item():.6f}")

        self.net.set_branches_mode(None)