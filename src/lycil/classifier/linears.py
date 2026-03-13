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
    """Linear classification head returning outputs as a dict.

    A thin wrapper around :class:`torch.nn.Linear` whose :meth:`forward`
    returns ``{"logits": ...}`` for API compatibility with other classifier
    heads in this library.

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Number of output classes.
        bias (bool, optional): If ``False``, the layer has no additive bias.
            (default: ``True``)
        device: Device for parameter allocation.
        dtype: Data type for parameter allocation.
    """

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
    def from_linear(
        cls, old_linear: nn.Linear, out_delta: int, in_delta: int = 0
    ) -> "SimpleLinear":
        """Create a SimpleLinear layer by expanding an existing linear layer.

        Args:
            old_linear (nn.Linear): The existing linear layer.
            out_delta (int): Number of new output features to add.
            in_delta (int): Number of new input features to add. (default: 0)

        Returns:
            SimpleLinear: The expanded SimpleLinear layer.

        """
        # head expansion from an existing linear
        new_linear = cls(
            in_features=old_linear.in_features + in_delta,
            out_features=old_linear.out_features + out_delta,
            bias=old_linear.bias is not None,
            device=old_linear.weight.device,
            dtype=old_linear.weight.dtype,
        )
        new_linear.weight[: old_linear.out_features, : old_linear.in_features].copy_(
            old_linear.weight
        )
        if old_linear.bias is not None:
            new_linear.bias[: old_linear.out_features].copy_(old_linear.bias)
        return new_linear


class CosineLinear(nn.Module):
    """Cosine similarity-based classification head with optional proxy support.

    Computes normalized dot products between L2-normalized input features and
    weight vectors. Supports multiple proxy vectors per class (reduced via
    :func:`reduce_proxies`) and an optional learnable scale parameter ``sigma``.

    Args:
        in_features (int): Size of each input feature vector.
        out_features (int): Number of output classes.
        num_proxy (int, optional): Number of proxy vectors per class.
            (default: ``1``)
        to_reduce (bool, optional): If ``True``, apply :func:`reduce_proxies`
            in :meth:`forward`. (default: ``False``)
        learn_scale (bool, optional): If ``True``, add a learnable scalar
            ``sigma`` that multiplies the output logits. (default: ``True``)
        device: Device for parameter allocation.
        dtype: Data type for parameter allocation.
    """

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
    """Cosine head split into frozen old-class and trainable new-class sub-heads.

    Used in incremental learning to keep old-class weights fixed while training
    new ones. :meth:`forward` returns per-group cosine scores and their
    concatenation, all scaled by a single learnable ``sigma``.

    Args:
        in_features (int): Size of each input feature vector.
        old_out_features (int): Number of old (already-seen) classes.
        new_out_features (int): Number of new classes added in the current task.
        num_proxy (int, optional): Number of proxy vectors per class.
            (default: ``1``)
        learn_scale (bool, optional): If ``True``, add a learnable scalar
            ``sigma`` shared across both sub-heads. (default: ``True``)
        device: Device for parameter allocation.
        dtype: Data type for parameter allocation.
    """

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



class SplitLinear(nn.Module):
    """Linear head split into frozen old-class and trainable new-class sub-heads.

    Used in incremental learning to keep old-class weights fixed while training
    new ones. :meth:`forward` returns per-group linear scores and their
    concatenation.

    Args:
        in_features (int): Size of each input feature vector.
        old_out_features (int): Number of old (already-seen) classes.
        new_out_features (int): Number of new classes added in the current task.
        bias (bool, optional): If ``False``, the layer has no additive bias.
            (default: ``True``)
        device: Device for parameter allocation.
        dtype: Data type for parameter allocation.
    """

    def __init__(
        self,
        in_features: int,
        old_out_features: int,
        new_out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = old_out_features + new_out_features
        self.old_out_features = old_out_features
        self.new_out_features = new_out_features

        self.old_head = SimpleLinear(
            in_features=in_features,
            out_features=old_out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.new_head = SimpleLinear(
            in_features=in_features,
            out_features=new_out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )


    @staticmethod
    @torch.no_grad()
    def _copy_linear_params(dst: nn.Linear, src: nn.Linear) -> None:
        """Copy parameters from one linear layer to another."""
        if dst.in_features != src.in_features or dst.out_features != src.out_features:
            raise ValueError(
                f"Shape mismatch: dst=({dst.out_features}, {dst.in_features}), "
                f"src=({src.out_features}, {src.in_features})"
            )

        dst.weight.copy_(src.weight)
        if dst.bias is not None and src.bias is not None:
            dst.bias.copy_(src.bias)
        elif (dst.bias is None) != (src.bias is None):
            raise ValueError("Bias configuration mismatch between dst and src.")

    @classmethod
    @torch.no_grad()
    def from_linear(
        cls,
        old_linear: nn.Linear,
        num_new: int,
    ) -> "SplitLinear":
        """Head expansion from an existing linear head.

        Args:
            old_linear (nn.Linear): The existing linear layer.
            num_new (int): Number of new classes to add.
            freeze_old (bool, optional): Whether to freeze old classifier.
                (default: ``True``)

        Returns:
            SplitLinear: Expanded split linear head.
        """
        new_head = cls(
            in_features=old_linear.in_features,
            old_out_features=old_linear.out_features,
            new_out_features=num_new,
            bias=old_linear.bias is not None,
            device=old_linear.weight.device,
            dtype=old_linear.weight.dtype,
        )
        cls._copy_linear_params(new_head.old_head, old_linear)

        return new_head

    @classmethod
    @torch.no_grad()
    def from_split_linear(
        cls,
        old_linear: "SplitLinear",
        num_new: int,
    ) -> "SplitLinear":
        """Head expansion from an existing SplitLinear head.

        Old classes in the new head will be:
            [old_linear.old_head classes] + [old_linear.new_head classes]

        and the newly added classes are initialized in ``new_head``.

        Args:
            old_linear (SplitLinear): Existing split linear head.
            num_new (int): Number of new classes to add.
            freeze_old (bool, optional): Whether to freeze merged old classifier.
                (default: ``True``)

        Returns:
            SplitLinear: Expanded split linear head.
        """
        bias = old_linear.old_head.bias is not None
        new_linear = cls(
            in_features=old_linear.in_features,
            old_out_features=old_linear.out_features,
            new_out_features=num_new,
            bias=bias,
            device=old_linear.old_head.weight.device,
            dtype=old_linear.old_head.weight.dtype,
        )

        old_n = old_linear.old_head.out_features
        new_n = old_linear.new_head.out_features

        # 把上一阶段的 old + new 都并到新的 old_head 里
        new_linear.old_head.weight[:old_n].copy_(old_linear.old_head.weight)
        new_linear.old_head.weight[old_n : old_n + new_n].copy_(
            old_linear.new_head.weight
        )

        if bias:
            assert new_linear.old_head.bias is not None
            assert old_linear.old_head.bias is not None
            assert old_linear.new_head.bias is not None

            new_linear.old_head.bias[:old_n].copy_(old_linear.old_head.bias)
            new_linear.old_head.bias[old_n : old_n + new_n].copy_(
                old_linear.new_head.bias
            )

        return new_linear

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        old_scores = self.old_head(x)["logits"]
        new_scores = self.new_head(x)["logits"]
        logits = torch.cat([old_scores, new_scores], dim=1)

        return {
            "old_scores": old_scores,
            "new_scores": new_scores,
            "logits": logits,
        }