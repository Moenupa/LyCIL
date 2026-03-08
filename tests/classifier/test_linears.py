import pytest
import torch
import torch.nn.functional as F

from lycil.classifier.linears import (
    CosineLinear,
    SimpleLinear,
    SplitCosineLinear,
    reduce_proxies,
)


@pytest.mark.runs_on(["cpu", "cuda", "npu"])
def test_simple_linear(device):
    in_features, out_features = 10, 5
    m = SimpleLinear(in_features, out_features, device=device)
    x = torch.randn(2, in_features, device=device)
    out = m(x)
    assert "logits" in out
    assert out["logits"].shape == (2, out_features)
    assert out["logits"].device.type == torch.device(device).type

    # Test expansion
    m_new = SimpleLinear.from_linear(m, out_delta=3)
    assert m_new.out_features == 8
    assert torch.allclose(m_new.weight[:out_features], m.weight)
    assert m_new.bias is not None
    assert m.bias is not None
    assert torch.allclose(m_new.bias[:out_features], m.bias)
    assert m_new.weight.device.type == torch.device(device).type


@pytest.mark.runs_on(["cpu", "cuda", "npu"])
def test_simple_linear_from_linear_without_bias(device):
    old = torch.nn.Linear(6, 4, bias=False, device=device)
    new = SimpleLinear.from_linear(old, out_delta=2)

    assert new.bias is None
    assert new.out_features == 6
    assert torch.allclose(new.weight[:4], old.weight)


@pytest.mark.runs_on(["cpu", "cuda", "npu"])
def test_cosine_linear(device):
    in_features, out_features, num_proxy = 10, 5, 2
    m = CosineLinear(
        in_features, out_features, num_proxy=num_proxy, to_reduce=True, device=device
    )
    x = torch.randn(2, in_features, device=device)
    out = m(x)
    assert out["logits"].shape == (2, out_features)
    assert out["logits"].device.type == torch.device(device).type


@pytest.mark.runs_on(["cpu", "cuda", "npu"])
def test_cosine_linear_matches_manual_forward(device):
    in_features, out_features = 4, 3
    x = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [2.0, -1.0, 0.5, 1.5]],
        device=device,
    )
    m = CosineLinear(
        in_features=in_features,
        out_features=out_features,
        num_proxy=1,
        to_reduce=False,
        learn_scale=True,
        device=device,
    )
    with torch.no_grad():
        m.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ],
                device=device,
            )
        )
        assert m.sigma is not None
        m.sigma.fill_(2.5)

    out = m(x)["logits"]
    expected = (
        F.linear(
            F.normalize(x, p=2, dim=1),
            F.normalize(m.weight, p=2, dim=1),
        )
        * 2.5
    )
    assert torch.allclose(out, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.runs_on(["cpu", "cuda", "npu"])
def test_cosine_linear_no_learned_scale(device):
    m = CosineLinear(8, 4, learn_scale=False, device=device)
    assert m.sigma is None


@pytest.mark.runs_on(["cpu", "cuda", "npu"])
def test_split_cosine_linear(device):
    in_features, old_out, new_out, num_proxy = 10, 5, 3, 2
    m = SplitCosineLinear(
        in_features, old_out, new_out, num_proxy=num_proxy, device=device
    )
    x = torch.randn(2, in_features, device=device)
    out = m(x)
    assert out["logits"].shape == (2, old_out + new_out)
    assert out["old_scores"].shape == (2, old_out)
    assert out["new_scores"].shape == (2, new_out)


@pytest.mark.runs_on(["cpu", "cuda", "npu"])
def test_split_cosine_linear_matches_manual_outputs(device):
    m = SplitCosineLinear(
        in_features=3,
        old_out_features=2,
        new_out_features=1,
        num_proxy=1,
        learn_scale=True,
        device=device,
    )
    x = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]], device=device)

    with torch.no_grad():
        m.old_head.weight.copy_(
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device)
        )
        m.new_head.weight.copy_(torch.tensor([[0.0, 0.0, 1.0]], device=device))
        assert m.sigma is not None
        m.sigma.fill_(1.75)

    out = m(x)
    old_expected = F.linear(
        F.normalize(x, p=2, dim=1), F.normalize(m.old_head.weight, p=2, dim=1)
    )
    new_expected = F.linear(
        F.normalize(x, p=2, dim=1), F.normalize(m.new_head.weight, p=2, dim=1)
    )
    logits_expected = torch.cat([old_expected, new_expected], dim=1) * 1.75

    assert torch.allclose(out["old_scores"], old_expected, atol=1e-6, rtol=1e-6)
    assert torch.allclose(out["new_scores"], new_expected, atol=1e-6, rtol=1e-6)
    assert torch.allclose(out["logits"], logits_expected, atol=1e-6, rtol=1e-6)


@pytest.mark.runs_on(["cpu", "cuda", "npu"])
def test_expansions(device):
    in_features, out_features = 10, 5
    cl = CosineLinear(in_features, out_features, num_proxy=1, device=device)

    # CosineLinear -> SplitCosineLinear
    scl = SplitCosineLinear.from_cosine_linear(cl, num_new=2)
    assert scl.out_features == 7
    assert torch.allclose(scl.old_head.weight, cl.weight)
    assert cl.sigma is not None
    assert scl.sigma is not None
    assert torch.allclose(scl.sigma, cl.sigma)

    # SplitCosineLinear -> SplitCosineLinear
    scl2 = SplitCosineLinear.from_split_cosine_linear(scl, num_new=3)
    assert scl2.out_features == 10
    # First 5 classes from original CL
    assert torch.allclose(scl2.old_head.weight[:5], cl.weight)
    # Next 2 classes from first SCL new_head
    assert torch.allclose(scl2.old_head.weight[5:7], scl.new_head.weight)


@pytest.mark.runs_on(["cpu", "cuda", "npu"])
def test_split_cosine_from_split_preserves_sigma(device):
    scl = SplitCosineLinear(5, 4, 2, num_proxy=1, learn_scale=True, device=device)
    with torch.no_grad():
        assert scl.sigma is not None
        scl.sigma.fill_(3.0)

    scl2 = SplitCosineLinear.from_split_cosine_linear(scl, num_new=3)
    assert scl2.sigma is not None
    assert torch.allclose(scl2.sigma, torch.tensor(3.0, device=device))


def test_reduce_proxies():
    # Test normal reduction
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])  # batch=1, out=4 (2 per class)
    res = reduce_proxies(logits, num_proxy=2)
    assert res.shape == (1, 2)
    expected = torch.tensor([[1.7311, 3.7311]])
    assert torch.allclose(res, expected, atol=1e-4, rtol=1e-4)

    # Test no reduction
    assert torch.all(reduce_proxies(logits, num_proxy=1) == logits)

    # Test error on invalid shape
    with pytest.raises(ValueError):
        reduce_proxies(torch.randn(1, 5), num_proxy=2)


def test_reduce_proxies_multi_batch_matches_manual_attention():
    logits = torch.tensor(
        [
            [1.0, 0.0, 2.0, 3.0],
            [0.5, 1.5, -1.0, -0.5],
        ]
    )
    out = reduce_proxies(logits, num_proxy=2)

    per_class = logits.view(2, 2, 2)
    attention = F.softmax(per_class, dim=-1)
    expected = (attention * per_class).sum(-1)
    assert torch.allclose(out, expected, atol=1e-6, rtol=1e-6)
