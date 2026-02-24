import math
from collections.abc import Callable
from typing import TypedDict

import torch
from torch import nn
from torch.nn import functional as F

LinearHead = Callable[[torch.Tensor], dict[str, torch.Tensor]]


class FactoryKwargs(TypedDict):
    device: str | int | torch.device | None
    dtype: torch.dtype | None


class SimpleLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:  # ty: ignore[invalid-method-override]
        return {"logits": super().forward(x)}

    @classmethod
    @torch.no_grad()
    def from_linear(cls, old_linear: nn.Linear, num_new: int) -> "SimpleLinear":
        """Create a SimpleLinear layer by expanding an existing linear layer.

        Args:
            old_linear (nn.Linear): The existing linear layer.
            num_new (int): Number of new output features to add.

        Returns:
            SimpleLinear: The expanded SimpleLinear layer.

        """
        # head expansion from an existing linear
        new_linear = cls(
            in_features=old_linear.in_features,
            out_features=old_linear.out_features + num_new,
            bias=old_linear.bias is not None,
            device=old_linear.weight.device,
            dtype=old_linear.weight.dtype,
        )
        new_linear.weight[: old_linear.out_features].copy_(old_linear.weight)
        if old_linear.bias is not None:
            new_linear.bias[: old_linear.out_features].copy_(old_linear.bias)
        return new_linear


class CosineLinear(nn.Module):
    """Cosine Linear layer with proxy support."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_proxy: int = 1,
        to_reduce: bool = False,
        learn_scale: bool = True,
        device=None,
        dtype=None,
    ):
        factory_kwargs: FactoryKwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_proxy = num_proxy
        self.to_reduce = to_reduce

        self.weight = nn.Parameter(
            torch.empty((out_features * num_proxy, in_features), **factory_kwargs)
        )

        if learn_scale:
            self.sigma = nn.Parameter(torch.tensor(1.0, **factory_kwargs))
        else:
            self.register_parameter("sigma", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters using a uniform distribution."""
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = F.linear(
            F.normalize(x, p=2, dim=1), F.normalize(self.weight, p=2, dim=1)
        )

        if self.to_reduce:
            logits = reduce_proxies(logits, self.num_proxy)

        if self.sigma is not None:
            logits = logits * self.sigma

        return {"logits": logits}


class SplitCosineLinear(nn.Module):
    """Split Cosine Linear layer for incremental learning."""

    def __init__(
        self,
        in_features: int,
        old_out_features: int,
        new_out_features: int,
        num_proxy: int = 1,
        learn_scale: bool = True,
        device=None,
        dtype=None,
    ):
        factory_kwargs: FactoryKwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.in_features = in_features
        self.out_features = old_out_features + new_out_features
        self.num_proxy = num_proxy

        self.old_head = CosineLinear(
            in_features,
            old_out_features,
            num_proxy=num_proxy,
            to_reduce=False,
            learn_scale=False,
            **factory_kwargs,
        )
        self.new_head = CosineLinear(
            in_features,
            new_out_features,
            num_proxy=num_proxy,
            to_reduce=False,
            learn_scale=False,
            **factory_kwargs,
        )

        if learn_scale:
            self.sigma = nn.Parameter(torch.tensor(1.0, **factory_kwargs))
        else:
            self.register_parameter("sigma", None)

    @classmethod
    @torch.no_grad()
    def from_cosine_linear(
        cls, old_linear: CosineLinear, num_new: int
    ) -> "SplitCosineLinear":
        """Head expansion, from an existing CosineLinear head.

        Args:
            old_linear (CosineLinear): The existing CosineLinear layer.
            num_new (int): Number of new output features to add.

        Returns:
            SplitCosineLinear: The expanded SplitCosineLinear layer.

        """
        new_head = cls(
            in_features=old_linear.in_features,
            old_out_features=old_linear.out_features,
            new_out_features=num_new,
            num_proxy=old_linear.num_proxy,
            learn_scale=old_linear.sigma is not None,
            device=old_linear.weight.device,
            dtype=old_linear.weight.dtype,
        )

        new_head.old_head.weight.copy_(old_linear.weight)
        if old_linear.sigma is not None:
            new_head.sigma.copy_(old_linear.sigma)
        return new_head

    @classmethod
    @torch.no_grad()
    def from_split_cosine_linear(
        cls, old_linear: "SplitCosineLinear", num_new: int
    ) -> "SplitCosineLinear":
        """Head expansion, from an existing SplitCosineLinear head.

        Args:
            old_linear (SplitCosineLinear): The existing SplitCosineLinear layer.
            num_new (int): Number of new output features to add.

        Returns:
            SplitCosineLinear: The expanded SplitCosineLinear layer.

        """
        new_linear = cls(
            in_features=old_linear.in_features,
            old_out_features=old_linear.out_features,
            new_out_features=num_new,
            num_proxy=old_linear.num_proxy,
            learn_scale=old_linear.sigma is not None,
            device=old_linear.old_head.weight.device,
            dtype=old_linear.old_head.weight.dtype,
        )
        old_classes_proxies = old_linear.old_head.out_features * old_linear.num_proxy
        new_linear.old_head.weight[:old_classes_proxies].copy_(
            old_linear.old_head.weight
        )
        new_linear.old_head.weight[old_classes_proxies:].copy_(
            old_linear.new_head.weight
        )
        if old_linear.sigma is not None:
            new_linear.sigma.copy_(old_linear.sigma)
        return new_linear

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        old_scores = self.old_head(x)["logits"]
        new_scores = self.new_head(x)["logits"]

        logits = torch.cat([old_scores, new_scores], dim=1)
        logits = reduce_proxies(logits, self.num_proxy)

        if self.sigma is not None:
            logits = logits * self.sigma

        return {
            "old_scores": reduce_proxies(old_scores, self.num_proxy),
            "new_scores": reduce_proxies(new_scores, self.num_proxy),
            "logits": logits,
        }


def reduce_proxies(logits: torch.Tensor, num_proxy: int) -> torch.Tensor:
    """Reduces proxy logits per class using attention-based weighting.

    Args:
        logits (torch.Tensor): The logits tensor of shape (batch_size, out_features).
        num_proxy (int): Number of proxies per class.

    Returns:
        torch.Tensor: Reduced logits tensor of shape (batch_size, num_classes).

    Raises:
        ValueError: If the number of output features is not divisible by the number of proxies.

    """
    if num_proxy == 1:
        return logits

    batch_size, out_features = logits.shape
    if out_features % num_proxy != 0:
        raise ValueError(
            f"Output features {out_features} not divisible by number of proxies ({num_proxy})."
        )

    num_classes = out_features // num_proxy
    simi_per_class = logits.view(batch_size, num_classes, num_proxy)
    attentions = F.softmax(simi_per_class, dim=-1)

    return (attentions * simi_per_class).sum(-1)
