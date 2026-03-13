import torch
import torch.nn as nn

from .linears import CosineLinear, SimpleLinear, SplitCosineLinear, SplitLinear

__all__ = [
    "CosineLinear",
    "SimpleLinear",
    "SplitCosineLinear",
    "SplitLinear",
    "expand_head",
    "make_head",
]

_CLASSIFIER_HEADS: dict[str, tuple[type[nn.Module], dict]] = {
    # key: (class, {optional kwargs})
    "linear": (SimpleLinear, {}),
    "split_linear": (SplitLinear, {}),
    "cosine": (CosineLinear, {"learn_scale": True}),
}


@torch.no_grad()
def make_head(
        in_features: int, out_features: int, head_type: str = "linear", **kwargs
) -> nn.Module:
    r"""Create a new classification head.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        head_type (str, optional): Type of the classification head. (default: "linear")
        kwargs: Override kwargs passed to the head constructor.

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


# @torch.no_grad()
# def expand_head(module: nn.Module, out_delta: int, in_delta: int = 0) -> nn.Module:
#     r"""Expand classifier to accommodate for more classes.  :math:`\texttt{out\_delta}=n_\text{after} - n_\text{before}\geq0`.
#
#     Args:
#         module (nn.Module): The classifier module to be expanded.
#         out_delta (int): A non-negative value for expanding newly added classes.
#         in_delta (int): A non-negative value for expanding features size.
#
#     Raises:
#         ValueError: If out_delta <= 0 or in_delta < 0.
#         NotImplementedError: If the classifier does not support expansion.
#
#     Returns:
#         nn.Module: The expanded classifier module.
#
#     """
#     if out_delta <= 0:
#         raise ValueError(f"Expanding for new heads {out_delta} must be >0.")
#     if in_delta < 0:
#         raise ValueError(f"Expanding for new features {in_delta} must be >=0.")
#
#     if isinstance(module, SimpleLinear):
#         new_linear = SimpleLinear.from_linear(module, out_delta, in_delta)
#         return new_linear
#
#     if in_delta > 0:
#         raise NotImplementedError(
#             "Input feature expansion not implemented for this head type."
#         )
#     if isinstance(module, CosineLinear):
#         new_linear = SplitCosineLinear.from_cosine_linear(module, out_delta)
#         new_linear.old_head.requires_grad_(False)
#         return new_linear
#
#     if isinstance(module, SplitCosineLinear):
#         new_linear = SplitCosineLinear.from_split_cosine_linear(module, out_delta)
#         new_linear.old_head.requires_grad_(False)
#         return new_linear
#
#     raise NotImplementedError(f"Classifier not expandable: {type(module)}.")


@torch.no_grad()
def expand_head(module: nn.Module, out_delta: int, in_delta: int = 0) -> nn.Module:
    r"""Expand classifier to accommodate more classes.

    :math:`\texttt{out\_delta}=n_\text{after} - n_\text{before}\geq0`.

    Args:
        module (nn.Module): The classifier module to be expanded.
        out_delta (int): A non-negative value for expanding newly added classes.
        in_delta (int): A non-negative value for expanding feature size.

    Raises:
        ValueError: If out_delta <= 0 or in_delta < 0.
        NotImplementedError: If the classifier does not support expansion.

    Returns:
        nn.Module: The expanded classifier module.
    """
    if out_delta <= 0:
        raise ValueError(f"Expanding for new heads {out_delta} must be > 0.")
    if in_delta < 0:
        raise ValueError(f"Expanding for new features {in_delta} must be >= 0.")

    # Simple linear head: support both class expansion and input feature expansion
    if isinstance(module, SimpleLinear):
        return SimpleLinear.from_linear(module, out_delta, in_delta)


    # Cosine heads currently do not support input feature expansion
    if in_delta > 0:
        raise NotImplementedError(
            "Input feature expansion not implemented for this head type."
        )

    if isinstance(module, SplitLinear):
        new_linear = SplitLinear.from_split_linear(module, out_delta)
        new_linear.old_head.requires_grad_(False)
        return new_linear

    if isinstance(module, CosineLinear):
        new_linear = SplitCosineLinear.from_cosine_linear(module, out_delta)
        new_linear.old_head.requires_grad_(False)
        return new_linear

    if isinstance(module, SplitCosineLinear):
        new_linear = SplitCosineLinear.from_split_cosine_linear(module, out_delta)
        new_linear.old_head.requires_grad_(False)
        return new_linear

    raise NotImplementedError(f"Classifier not expandable: {type(module)}.")