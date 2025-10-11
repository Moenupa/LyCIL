from typing import List, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
import torchvision.transforms as T
import torchvision.datasets as dsets
import numpy as np
import lightning.pytorch as pl
from .buffer import ExemplarBuffer, ExemplarDataset

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

class _Normalize(T.Normalize):
    # Convenience wrapper for readability
    pass

def _make_transforms(image_size: int = 32):
    train_tf = T.Compose([
        T.RandomCrop(image_size, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        _Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    test_tf = T.Compose([
        T.ToTensor(),
        _Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    return train_tf, test_tf

class CIFAR100DataModule(pl.LightningDataModule):
    """CIFAR-100 incremental learning DataModule with exemplar rehearsal mixing."""
    def __init__(
        self,
        root: str = "./data",
        download: bool = True,
        increment: int = 10,
        batch_size: int = 128,
        num_workers: int = 4,
        class_order: Optional[List[int]] = None,
        seed: int = 0,
    ):
        super().__init__()
        self.root = root
        self.download = download
        self.increment = int(increment)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.class_order = class_order
        self.seed = int(seed)

        self.train_tf, self.test_tf = _make_transforms(32)

        # Stateful across tasks
        self._dataset_train = None
        self._dataset_test = None
        self.task_id = 0
        self.seen_classes: List[int] = []
        self.current_classes: List[int] = []
        self.num_classes_total = 100

        self.buffer: Optional[ExemplarBuffer] = None  # will be set by trainer loop

    def prepare_data(self):
        dsets.CIFAR100(self.root, train=True, download=self.download)
        dsets.CIFAR100(self.root, train=False, download=self.download)

    def setup(self, stage: Optional[str] = None):
        full_train = dsets.CIFAR100(self.root, train=True, transform=self.train_tf, download=False)
        full_test = dsets.CIFAR100(self.root, train=False, transform=self.test_tf, download=False)

        if self.class_order is None:
            rng = np.random.RandomState(self.seed)
            order = list(range(100))
            rng.shuffle(order)
            self.class_order = order

        self._full_train = full_train
        self._full_test = full_test

    # ----- task control -----
    def num_tasks(self) -> int:
        return int(np.ceil(self.num_classes_total / self.increment))

    def set_task_id(self, task_id: int):
        self.task_id = int(task_id)
        start = task_id * self.increment
        end = min((task_id + 1) * self.increment, self.num_classes_total)
        self.current_classes = self.class_order[start:end]
        self.seen_classes = sorted(self.class_order[:end])

        # Filter datasets
        self._dataset_train = self._filter_dataset(self._full_train, self.current_classes, train=True)
        self._dataset_test = self._filter_dataset(self._full_test, self.seen_classes, train=False)

    def _filter_dataset(self, dataset: dsets.CIFAR100, cls_ids: List[int], train: bool) -> Dataset:
        idx = [i for i, y in enumerate(dataset.targets) if y in cls_ids]
        return Subset(dataset, idx)

    # ----- rehearsal mixing -----
    def train_dataloader(self):
        if self.buffer is not None and len(self.buffer) > 0:
            mix = ConcatDataset([self._dataset_train, self.buffer.make_dataset()])
        else:
            mix = self._dataset_train
        return DataLoader(mix, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self._dataset_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return self.val_dataloader()

    def build_class_loader(self, class_id: int, train: bool, batch_size: int, shuffle: bool, num_workers: int):
        """Create a loader that yields only a single class from train or test set (used for herding)."""
        base = self._full_train if train else self._full_test
        idx = [i for i, y in enumerate(base.targets) if y == class_id]
        subset = Subset(base, idx)
        return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
