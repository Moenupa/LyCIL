from collections.abc import Callable

import numpy as np
import PIL.Image
import torch
import torchvision.transforms as T
from datasets.formatting import TorchFormatter, _register_formatter

_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD = (0.2470, 0.2435, 0.2616)

_CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
_CIFAR100_STD = (0.2675, 0.2565, 0.2761)

_IMAGENET1K_MEAN = (0.485, 0.456, 0.406)
_IMAGENET1K_STD = (0.229, 0.224, 0.225)


def get_transforms(name: str) -> tuple["Callable", "Callable"]:
    """Return train and test transforms for a named dataset preset.

    Args:
        name (str): Transform preset name. Supported values: ``"cifar10"``,
            ``"cifar100"``.

    Returns:
        tuple[Callable, Callable]: ``(train_transform, test_transform)`` pair of
            :class:`~torchvision.transforms.Compose` objects.

    Raises:
        ValueError: If ``name`` is not a recognized preset.
    """
    match name.lower():
        case "cifar10":
            train_tf = T.Compose(
                [
                    T.RandomCrop(32, padding=4),
                    T.RandomHorizontalFlip(),
                    T.ColorJitter(brightness=63 / 255),
                    T.ToTensor(),
                    T.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
                ]
            )
            test_tf = T.Compose(
                [
                    T.ToTensor(),
                    T.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
                ]
            )
            return train_tf, test_tf
        case "cifar100":
            train_tf = T.Compose(
                [
                    T.RandomCrop(32, padding=4),
                    T.RandomHorizontalFlip(),
                    T.ColorJitter(brightness=63 / 255),
                    T.ToTensor(),
                    T.Normalize(_CIFAR100_MEAN, _CIFAR100_STD),
                ]
            )
            test_tf = T.Compose(
                [
                    T.ToTensor(),
                    T.Normalize(_CIFAR100_MEAN, _CIFAR100_STD),
                ]
            )
            return train_tf, test_tf
        case "imagenet-1k" | "imagenet1k" | "imagenet" | "ilsvrc2012":
            train_tf = T.Compose(
                [
                    T.RandomResizedCrop(224),
                    T.RandomHorizontalFlip(),
                    T.ColorJitter(brightness=63 / 255),
                    T.ToTensor(),
                    T.Normalize(_IMAGENET1K_MEAN, _IMAGENET1K_STD),
                ]
            )
            test_tf = T.Compose(
                [
                    T.Resize(256),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    T.Normalize(_IMAGENET1K_MEAN, _IMAGENET1K_STD),
                ]
            )
            return train_tf, test_tf
        case "imagenet-100" | "imagenet100":
            train_tf = T.Compose(
                [
                    T.RandomResizedCrop(224),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize(_IMAGENET1K_MEAN, _IMAGENET1K_STD),
                ]
            )
            test_tf = T.Compose(
                [
                    T.Resize(256),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    T.Normalize(_IMAGENET1K_MEAN, _IMAGENET1K_STD),
                ]
            )
            return train_tf, test_tf
        case "tiny-imagenet" | "tiny-imagenet-200" | "tinyimagenet" | "tinyimagenet200":
            train_tf = T.Compose(
                [
                    T.RandomResizedCrop(64),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize(_IMAGENET1K_MEAN, _IMAGENET1K_STD),
                ]
            )
            test_tf = T.Compose(
                [
                    T.Resize(64),
                    T.CenterCrop(64),
                    T.ToTensor(),
                    T.Normalize(_IMAGENET1K_MEAN, _IMAGENET1K_STD),
                ]
            )
            return train_tf, test_tf
        case _:
            _support_msg = (
                "We support most common datasets and aliases to our best effort,"
                + "but you may have a custom dataset or a uncommon alias."
                + "Consider raising an issue https://github.com/Moenupa/LyCIL/issues."
            )
            raise ValueError(f"Unknown dataset/transform: {name}. {_support_msg}")


def register_tf_as_formatter(name: str) -> None:
    """Register HuggingFace dataset formatters for train and test transforms.

    Creates two named formatters, ``"{name}_train"`` and ``"{name}_test"``,
    by calling :func:`_register_custom_formatter` for each transform variant.

    Args:
        name (str): Transform preset name (e.g., ``"cifar10"``).

    Raises:
        ValueError: If ``name`` is not a recognized transform preset.
    """
    train_tf, test_tf = get_transforms(name)

    _register_custom_formatter(train_tf, f"{name}_train")
    _register_custom_formatter(test_tf, f"{name}_test")


def _register_custom_formatter(
    transform: "Callable[[PIL.Image.Image], torch.Tensor]",
    name: str,
    aliases: list[str] | None = None,
):
    """Register a HuggingFace dataset formatter that applies a PIL transform.

    The registered formatter intercepts PIL images before they are converted to
    tensors and applies ``transform`` in-place, then falls back to the standard
    :class:`~datasets.formatting.TorchFormatter` for all other value types.

    Args:
        transform (Callable[[PIL.Image.Image], torch.Tensor]): Transform applied
            to each PIL image before tensor conversion. Example::

                import torchvision.transforms as T

                transform = T.Compose(
                    [
                        T.RandomCrop(32, padding=4),
                        T.RandomHorizontalFlip(),
                        T.ToTensor(),
                        T.Normalize(mean, std),
                    ]
                )

        name (str): Formatter name used with :meth:`~datasets.Dataset.set_format`.
        aliases (list[str] | None, optional): Additional aliases for the same
            formatter. (default: ``None``)
    """

    # injects the transform into the formatter
    # before PILImage -> Tensor conversion
    class CustomFormatter(TorchFormatter):
        def _tensorize(self, value):
            """Zero/low-copy tensor conversion with smart dtype handling."""
            # Fast path for strings, bytes, None
            if isinstance(value, (str, bytes, type(None))):
                return value

            # Handle string arrays
            if isinstance(value, (np.character, np.ndarray)) and np.issubdtype(
                value.dtype, np.character
            ):
                return value.tolist()

            # skipped PIL check because we include datasets[vision] as a dep
            # if config.PIL_AVAILABLE and "PIL" in sys.modules:

            if isinstance(value, PIL.Image.Image):
                # NEW: inject transform here, which goes before tensor conversion
                # we assume transform is a PILImage -> torch.Tensor transform
                tensor: torch.Tensor = transform(value)
                assert isinstance(tensor, torch.Tensor)
                return tensor

                # BACKUP: original datasets.TorchFormatter, never reached
                # Single conversion path: PIL -> numpy -> torch
                arr = np.asarray(value)
                if arr.ndim == 2:
                    arr = arr[:, :, np.newaxis]
                # Use moveaxis instead of transpose
                arr = np.moveaxis(arr, -1, 0)  # HWC -> CHW
                # Ensure contiguous for zero-copy conversion
                if not arr.flags.c_contiguous:
                    arr = np.ascontiguousarray(arr)
                # Ensure array is writable for torch conversion
                if not arr.flags.writeable:
                    arr = arr.copy()
                return torch.from_numpy(arr)

            # fallback to datasets.TorchFormatter
            return super()._tensorize(value)

    _register_formatter(CustomFormatter, name, aliases)

    return None
