from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import numpy as np
import PIL.Image
import torch
import torchvision.transforms as T

from ..constants import _X_COLUMN_NAME

if TYPE_CHECKING:
    from datasets import Dataset

_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD = (0.2470, 0.2435, 0.2616)

_CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
_CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def get_transforms(name: str) -> tuple["Callable", "Callable"]:
    """Return train and test transforms for dataset name.

    Raises:
        name (str): dataset name. Supported values: ``"cifar10"``,
            ``"cifar100"``.

    Returns:
         tuple[Callable, Callable]: ``(train_transform, test_transform)``
        pair of :class:`~torchvision.transforms.Compose` objects.

    Raises:
        ValueError: If ``name`` is not a recognized preset.
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
            # train_tf = T.Compose(
            #     [
            #         T.RandomCrop(32, padding=4, padding_mode="reflect"),
            #         T.RandomHorizontalFlip(),
            #         T.AutoAugment(policy=T.AutoAugmentPolicy.CIFAR10),
            #         T.ToTensor(),
            #         T.Normalize(_CIFAR100_MEAN, _CIFAR100_STD),
            #         T.RandomErasing(
            #             p=0.25,
            #             scale=(0.02, 0.2),
            #             ratio=(0.3, 3.3),
            #             value="random",
            #         ),
            #     ]
            # )
            test_tf = T.Compose(
                [
                    T.ToTensor(),
                    T.Normalize(_CIFAR100_MEAN, _CIFAR100_STD),
                ]
            )
            return train_tf, test_tf
        case _:
            raise ValueError(f"Unknown transform name: {name}")



def get_transform(
    name: str,
    mode: Literal["train", "test"] = "train",
) -> "Callable[[PIL.Image.Image], torch.Tensor]":
    """Resolve a torchvision transform by dataset name and mode."""

    train_tf, test_tf = get_transforms(name)
    if mode == "train":
        return train_tf
    if mode == "test":
        return test_tf
    raise ValueError(f"Unknown transform mode: {mode}")



def _to_pil_image(image) -> PIL.Image.Image:
    if isinstance(image, PIL.Image.Image):
        return image
    if isinstance(image, np.ndarray):
        return PIL.Image.fromarray(image)
    raise TypeError(
        "Expected image value to be a PIL image or numpy array, "
        + f"got {type(image)}."
    )



def make_hf_transform(
    transform: "Callable[[PIL.Image.Image], torch.Tensor]",
    image_column_name: str = _X_COLUMN_NAME,
) -> "Callable[[dict], dict]":
    """Wrap a torchvision-style image transform for ``datasets.Dataset.set_transform``.

    The returned callable only transforms the image column and leaves the other columns
    to Hugging Face Datasets via ``output_all_columns=True``.
    """

    def _apply_one(image) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            return image
        return transform(_to_pil_image(image))

    def _hf_transform(batch: dict) -> dict:
        images = batch[image_column_name]
        if isinstance(images, list):
            return {image_column_name: [_apply_one(image) for image in images]}
        return {image_column_name: _apply_one(images)}

    return _hf_transform



def apply_dataset_transform(
    dataset: "Dataset",
    transform: "Callable[[PIL.Image.Image], torch.Tensor] | None" = None,
    image_column_name: str = _X_COLUMN_NAME,
) -> "Dataset":
    """Apply runtime formatting to a Hugging Face dataset.

    - If ``transform`` is provided, use ``set_transform`` for lazy image preprocessing.
    - Otherwise, fall back to ``set_format('torch')`` to preserve the previous behavior.
    """

    dataset.reset_format()
    if transform is None:
        dataset.set_format("torch")
        return dataset

    dataset.set_transform(
        make_hf_transform(transform, image_column_name=image_column_name),
        columns=[image_column_name],
        output_all_columns=True,
    )
    return dataset
