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


def get_transforms(name: str) -> tuple["Callable", "Callable"]:
    """Get train and test transforms according to common alias ``name``.

    Raises:
        ValueError: Unknown ``name``

    Returns:
        tuple[T.Compose, T.Compose]: train_transform, test_transform

    """
    match name:
        case "cifar10":
            train_tf = T.Compose(
                [
                    T.RandomCrop(32, padding=4),
                    T.RandomHorizontalFlip(),
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
        case _:
            raise ValueError(f"Unknown transform name: {name}")


def register_tf_as_formatter(name: str) -> None:
    """Register custom formatter with transforms for the given ``name``.

    Raises:
        ValueError: Unknown ``name``

    Args:
        name (str): Name of the transform set to register.

    """
    train_tf, test_tf = get_transforms(name)

    _register_custom_formatter(train_tf, f"{name}_train")
    _register_custom_formatter(test_tf, f"{name}_test")


def _register_custom_formatter(
    transform: "Callable[[PIL.Image.Image], torch.Tensor]",
    name: str,
    aliases: list[str] | None = None,
):
    """Register a custom formatter that applies the given transform `transform` before converting PIL images to torch tensors.

    Args:
        transform (Callable[[PIL.Image.Image], torch.Tensor]): A transform function that takes a
            PIL Image and returns a torch Tensor. E.g.,
            ```py
            import torchvision.transforms as T

            transform = T.Compose(
                [
                    T.RandomCrop(32, padding=4),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
                ]
            )
            ```
        name (str): Name of the formatter to register.
        aliases (list[str] | None, optional): Extra aliases for the formatter. (default: None)

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
