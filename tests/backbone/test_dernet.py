import pytest
import torch

from lycil.backbone import ConvNetArgs, DERNetBackbone


@pytest.mark.runs_on(["cpu", "cuda", "npu"])
def test_dernet_out_dim_raises_before_initialization(device):
    model = DERNetBackbone(ConvNetArgs(name="resnet18", pretrained=False)).to(device)

    with pytest.raises(RuntimeError, match="not initialized yet"):
        _ = model.out_dim


@pytest.mark.runs_on(["cpu", "cuda", "npu"])
def test_dernet_prepare_for_new_task_initializes_convnet(device):
    model = DERNetBackbone(ConvNetArgs(name="resnet18", pretrained=False)).to(device)
    model.prepare_for_new_task()
    model.to(device)

    assert len(model.convnets) == 1
    assert model.out_dim == 512
    assert model.feature_dim == 512

    x = torch.randn(2, 3, 32, 32, device=device)
    out = model.forward_layerwise(x)
    assert set(out.keys()) == {"features"}
    assert out["features"].shape == (2, 1000)


@pytest.mark.runs_on(["cpu", "cuda", "npu"])
def test_dernet_prepare_for_new_task_copies_previous_weights(device):
    model = DERNetBackbone(ConvNetArgs(name="resnet18", pretrained=False)).to(device)
    model.prepare_for_new_task()
    model.to(device)

    with torch.no_grad():
        for p in model.convnets[0].parameters():
            p.fill_(0.125)

    model.prepare_for_new_task()
    model.to(device)
    assert len(model.convnets) == 2

    old_state = model.convnets[0].state_dict()
    new_state = model.convnets[1].state_dict()
    for key in old_state:
        assert torch.allclose(old_state[key], new_state[key])


@pytest.mark.runs_on(["cpu", "cuda", "npu"])
def test_dernet_feature_dim_and_forward_after_multiple_tasks(device):
    model = DERNetBackbone(ConvNetArgs(name="resnet18", pretrained=False)).to(device)
    model.prepare_for_new_task()
    model.prepare_for_new_task()
    model.to(device)

    assert model.out_dim == 512
    assert model.feature_dim == 1024

    x = torch.randn(2, 3, 32, 32, device=device)
    features = model(x)
    assert features.shape == (2, 2000)
