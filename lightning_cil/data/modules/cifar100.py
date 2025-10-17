from typing import Optional

import torchvision.transforms as T
from numpy.random import RandomState
from torchvision.datasets import CIFAR100

from ..cil_datamodule import BaseCILDataModule, CILDataset

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
    """
    Data module for CIFAR100.

    Args:
        root (str, optional): root directory of the dataset. Default: "./data".
        download (bool, optional): whether to download data. Default: True.
        num_class_per_task (int, optional): increment per task. Default: 10.
        batch_size (int, optional): batch size. Default: 64.
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

        # no dup in class_order
        if class_order is None:
            index2target = RandomState(seed).permutation(self.num_class_total)
        else:
            assert len(class_order) == self.num_class_total, (
                f"{self.__class__.__name__}: class_order expects {self.num_class_total} entries, got {class_order}"
            )
            index2target = class_order
        self._target2index = {
            target: order_index for order_index, target in enumerate(index2target)
        }
        self.target_transform = lambda x: self._target2index[x]

        self.set_task(0)

        self.train_tf, self.test_tf = _make_transforms(32)
        self.buffer = None  # will be set by trainer loop

    def prepare_data(self):
        CIFAR100(self.root, train=True, download=self.download)
        CIFAR100(self.root, train=False, download=self.download)

    def setup(self, stage: Optional[str] = None):
        # target transformed to fake index
        self.cifar100_train = CIFAR100(
            self.root,
            train=True,
            transform=self.train_tf,
            target_transform=self.target_transform,
            download=False,
        )
        self.cifar100_test = CIFAR100(
            self.root,
            train=False,
            transform=self.test_tf,
            target_transform=self.target_transform,
            download=False,
        )

    def train_dataloader(self):
        return self.build_loader(
            CILDataset(
                BaseCILDataModule.dataset_by_target(
                    self.cifar100_train, self.classes_current
                ),
                buffer=self.buffer,
            ),
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return self.build_loader(
            BaseCILDataModule.dataset_by_target(self.cifar100_test, self.classes_seen),
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        return self.val_dataloader()
