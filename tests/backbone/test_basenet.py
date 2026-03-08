import pytest
import torch
from torch import nn
from torchvision.models import resnet

from lycil.backbone.basenet import BaseBackbone, ConvNetArgs, get_convnet


@pytest.mark.runs_on(["cpu", "cuda", "npu"])
def test_convnet_args_defaults(device):
    args = ConvNetArgs()
    assert args.name == "resnet18"
    assert args.pretrained is False
    assert args.cifar is False


@pytest.mark.runs_on(["cpu", "cuda", "npu"])
@pytest.mark.parametrize(
    ("name", "expected_dim", "expected_type"),
    [
        ("resnet18", 512, resnet.ResNet),
        ("resnet34", 512, resnet.ResNet),
        ("resnet50", 2048, resnet.ResNet),
    ],
)
def test_get_convnet_variants_return_expected_shapes(
    device, name: str, expected_dim: int, expected_type: type[nn.Module]
):
    net, feat_dim = get_convnet(
        ConvNetArgs(name=name, pretrained=False)  # ty: ignore[invalid-argument-type]
    )

    assert isinstance(net, expected_type)
    assert feat_dim == expected_dim

    x = torch.randn(2, 3, 32, 32)
    out = net(x)
    assert out.shape == (2, 1000)


@pytest.mark.runs_on(["cpu", "cuda", "npu"])
def test_get_convnet_resnet50_cifar_stem_overrides(device):
    net, feat_dim = get_convnet(ConvNetArgs(name="resnet50", cifar=True))

    assert feat_dim == 2048
    assert isinstance(net.conv1, nn.Conv2d)
    assert net.conv1.kernel_size == (3, 3)
    assert net.conv1.stride == (1, 1)
    assert isinstance(net.maxpool, nn.Identity)
    assert isinstance(net.fc, nn.Identity)

    x = torch.randn(2, 3, 32, 32)
    out = net(x)
    assert out.shape == (2, 2048)


@pytest.mark.runs_on(["cpu", "cuda", "npu"])
def test_get_convnet_raises_on_unknown_variant(device):
    bad_args = ConvNetArgs(name="not-a-resnet")  # ty: ignore[invalid-argument-type]

    with pytest.raises(ValueError, match="Unknown ResNet variant"):
        get_convnet(bad_args)


class _ToyBackbone(BaseBackbone):
    def __init__(self):
        super().__init__(ConvNetArgs(name="resnet18"))

    def forward_layerwise(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"features": x}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    @property
    def out_dim(self) -> int:
        return 4

    @property
    def feature_dim(self) -> int:
        return 4


@pytest.mark.runs_on(["cpu", "cuda", "npu"])
def test_base_backbone_subclass_contract(device):
    model = _ToyBackbone()
    x = torch.randn(3, 4)
    assert torch.allclose(model(x), x)
    assert torch.allclose(model.forward_layerwise(x)["features"], x)
    assert model.out_dim == 4
    assert model.feature_dim == 4
