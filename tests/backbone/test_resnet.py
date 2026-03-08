import pytest
import torch

from lycil.backbone import ConvNetArgs, ResNetBackbone


@pytest.mark.runs_on(["cpu", "cuda", "npu"])
def test_resnet_backbone_feature_dimensions(device):
    model = ResNetBackbone(ConvNetArgs(name="resnet18", pretrained=False)).to(device)
    assert model.out_dim == 512
    assert model.feature_dim == 512


@pytest.mark.runs_on(["cpu", "cuda", "npu"])
@pytest.mark.parametrize(
    ("name", "expected_dim"),
    [("resnet18", 512), ("resnet34", 512), ("resnet50", 2048)],
)
def test_resnet_backbone_forward_layerwise_shapes(device, name: str, expected_dim: int):
    model = ResNetBackbone(
        ConvNetArgs(name=name, pretrained=False)  # ty: ignore[invalid-argument-type]
    ).to(device)
    x = torch.randn(2, 3, 32, 32, device=device)

    out = model.forward_layerwise(x)
    assert set(out.keys()) == {"l1", "l2", "l3", "l4", "features"}
    assert out["features"].shape == (2, expected_dim)
    assert out["l1"].shape[0] == 2
    assert out["l2"].shape[0] == 2
    assert out["l3"].shape[0] == 2
    assert out["l4"].shape[0] == 2


@pytest.mark.runs_on(["cpu", "cuda", "npu"])
def test_resnet_backbone_forward_matches_layerwise_features(device):
    model = ResNetBackbone(ConvNetArgs(name="resnet18", pretrained=False)).to(device)
    x = torch.randn(2, 3, 32, 32, device=device)

    direct = model(x)
    layerwise = model.forward_layerwise(x)["features"]
    assert torch.allclose(direct, layerwise, atol=1e-6, rtol=1e-6)
