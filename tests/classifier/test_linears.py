import pytest
import torch

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
    m_new = SimpleLinear.from_linear(m, num_new=3)
    assert m_new.out_features == 8
    assert torch.allclose(m_new.weight[:out_features], m.weight)


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
def test_expansions(device):
    in_features, out_features = 10, 5
    cl = CosineLinear(in_features, out_features, num_proxy=1, device=device)

    # CosineLinear -> SplitCosineLinear
    scl = SplitCosineLinear.from_cosine_linear(cl, num_new=2)
    assert scl.out_features == 7
    assert torch.allclose(scl.old_head.weight, cl.weight)

    # SplitCosineLinear -> SplitCosineLinear
    scl2 = SplitCosineLinear.from_split_cosine_linear(scl, num_new=3)
    assert scl2.out_features == 10
    # First 5 classes from original CL
    assert torch.allclose(scl2.old_head.weight[:5], cl.weight)
    # Next 2 classes from first SCL new_head
    assert torch.allclose(scl2.old_head.weight[5:7], scl.new_head.weight)


def test_reduce_proxies():
    # Test normal reduction
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])  # batch=1, out=4 (2 per class)
    res = reduce_proxies(logits, num_proxy=2)
    assert res.shape == (1, 2)

    # Test no reduction
    assert torch.all(reduce_proxies(logits, num_proxy=1) == logits)

    # Test error on invalid shape
    with pytest.raises(ValueError):
        reduce_proxies(torch.randn(1, 5), num_proxy=2)
