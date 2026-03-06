import torch
import torch.nn as nn

from .linears import CosineLinear, SimpleLinear, SplitCosineLinear

__all__ = [
    "make_head",
    "expand_head",
    "CosineClassifier",
    "SimpleLinear",
    "SplitCosineLinear",
]


_CLASSIFIER_HEADS: dict[str, tuple[type[nn.Module], dict]] = {
    # key: (class, {optional kwargs})
    "linear": (SimpleLinear, {}),
    # "cosine": (CosineLinear, {"num_proxy": 10, "to_reduce": True, "learn_scale": True}),
    "cosine": (CosineLinear, {"num_proxy": 10, "to_reduce": True, "learn_scale": False}),
}


@torch.no_grad()
def make_head(
    in_features: int, out_features: int, head_type: str = "linear", **kwargs
) -> nn.Module:
    r"""Create a new classification head.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.

    Raises:
        ValueError: If head_type is not supported.

    Returns:
        nn.Module: The classification head.
    """
    if head_type not in _CLASSIFIER_HEADS:
        raise ValueError(
            f"Unknown head type `{head_type}`. Candidates: {_CLASSIFIER_HEADS.keys()}"
        )

    cls, default_kwargs = _CLASSIFIER_HEADS[head_type]
    # incoming kwargs override default_kwargs
    return cls(in_features, out_features, **(default_kwargs | kwargs))


@torch.no_grad()
def expand_head(module: nn.Module, num_new: int) -> nn.Module:
    r"""Expand classifier to accommodate for more classes.  :math:`\texttt{num\_new}=n_\text{after} - n_\text{before}\geq0`.

    Args:
        module (nn.Module): The classifier module to be expanded.
        num_new (int): A non-negative value for number of newly added classes.

    Raises:
        ValueError: If num_new <= 0.
        NotImplementedError: If the classifier does not support expansion.

    Returns:
        nn.Module: The expanded classifier module.
    """
    if num_new <= 0:
        raise ValueError(f"Expanding for new heads {num_new} must be >0.")

    if isinstance(module, SimpleLinear):
        new_linear = SimpleLinear.from_linear(module, num_new)
        return new_linear

    if isinstance(module, CosineLinear):
        new_linear = SplitCosineLinear.from_cosine_linear(module, num_new)
        new_linear.old_head.requires_grad_(False)
        new_linear.new_head.requires_grad_(True)
        return new_linear

    if isinstance(module, SplitCosineLinear):
        new_linear = SplitCosineLinear.from_split_cosine_linear(module, num_new)
        new_linear.old_head.requires_grad_(False)
        new_linear.new_head.requires_grad_(True)
        return new_linear

    raise NotImplementedError(f"Classifier not expandable: {type(module)}.")
