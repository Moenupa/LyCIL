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


class CIFAR100DataModule(BaseCILDataModule):
    def __init__(
        self,
        root: str = "./data",
        download: bool = True,
        num_class_per_task: int = 10,
        batch_size: int = 128,
        num_workers: int = 4,
        class_order: list[int] | None = None,
        seed: int = 0,
    ):
        super().__init__()
        self.root = root
        self.download = download
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.num_class_total = 100
        self.num_class_per_task = num_class_per_task
        self.seed = seed
        self.class_order = class_order or list(range(self.num_class_total))
        if class_order is None:
            np.random.RandomState(self.seed).shuffle(self.class_order)

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

    def _filter_dataset(self, dataset: CIFAR100, class_ids: list[int]) -> Dataset:
        idx = [i for i, (_, label) in enumerate(dataset) if label in class_ids]
        return Subset(dataset, idx)

    @property
    def _dataset_train(self) -> Dataset:
        return self._filter_dataset(self.cifar100_train, self.classes_current)

    @property
    def _dataset_test(self) -> Dataset:
        return self._filter_dataset(self.cifar100_test, self.classes_seen)

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
