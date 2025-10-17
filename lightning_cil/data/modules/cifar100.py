from typing import Optional

import numpy as np
import torchvision.transforms as T
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision.datasets import CIFAR100

from ..cil_datamodule import BaseCILDataModule

_CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
_CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def _make_transforms(image_size: int = 32):
    train_tf = T.Compose(
        [
            T.RandomCrop(image_size, padding=4),
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


def _filter_dataset_by_classid(
    dataset: CIFAR100, class_idx: list[int], class_order: list[int | str]
) -> Dataset:
    selected_targets = set(class_order[i] for i in class_idx)
    idx = [i for i, (_, t) in enumerate(dataset) if t in selected_targets]
    return Subset(dataset, idx)


class CIFAR100DataModule(BaseCILDataModule):
    """
    Data module for CIFAR100.

    Args:
        root (str, optional): root directory of the dataset. Default: "./data".
        download (bool, optional): whether to download data. Default: True.
        num_class_per_task (int, optional): increment per task. Default: 10.
        batch_size (int, optional): batch size. Default: 128.
        num_workers (int, optional): number of workers. Default: 4.
        class_order (list[int] | None, optional): list of orders, see :ref:`BaseCILDataModule`. Default: None.
        seed (int, optional): random seed. Default: 0.
    """

    def __init__(
        self,
        root: str = "./data",
        download: bool = True,
        num_class_per_task: int = 10,
        batch_size: int = 64,
        num_workers: int = 4,
        class_order: list[int] | None = None,
        seed: int = 0,
    ):
        super().__init__(
            root=root,
            num_class_per_task=num_class_per_task,
            num_class_total=100,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        self.download = download

        self.class_order = class_order or list(range(self.num_class_total))
        if class_order is None:
            np.random.RandomState(seed).shuffle(self.class_order)

        self.set_task(0)

        self.train_tf, self.test_tf = _make_transforms(32)
        self.buffer = None  # will be set by trainer loop

    def prepare_data(self):
        CIFAR100(self.root, train=True, download=self.download)
        CIFAR100(self.root, train=False, download=self.download)

    def setup(self, stage: Optional[str] = None):
        self.cifar100_train = CIFAR100(
            self.root, train=True, transform=self.train_tf, download=False
        )
        self.cifar100_test = CIFAR100(
            self.root, train=False, transform=self.test_tf, download=False
        )

    @property
    def _dataset_train(self) -> Dataset:
        return _filter_dataset_by_classid(
            self.cifar100_train, self.classes_current, self.class_order
        )

    @property
    def _dataset_test(self) -> Dataset:
        return _filter_dataset_by_classid(
            self.cifar100_test, self.classes_seen, self.class_order
        )

    def train_dataloader(self):
        if self.buffer is not None and len(self.buffer) > 0:
            mix = ConcatDataset([self._dataset_train, self.buffer.make_dataset()])
        else:
            mix = self._dataset_train
        assert len(mix) > 0, (
            f"{self.__class__.__name__}: no training data for task {self.task_id} with classes {self.classes_current}"
        )

        return DataLoader(
            mix,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self._dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return self.val_dataloader()
