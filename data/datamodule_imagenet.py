from typing import List, Optional
import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
import torchvision.transforms as T
import torchvision.datasets as dsets
import numpy as np
import lightning.pytorch as pl
from .buffer import ExemplarBuffer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def _make_transforms(image_size: int = 224):
    train_tf = T.Compose([
        T.RandomResizedCrop(image_size),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    test_tf = T.Compose([
        T.Resize(256),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, test_tf

class ImageNetDataModule(pl.LightningDataModule):
    """ImageNet incremental DataModule expecting ImageFolder structure.

    root/
      train/<class>/*.JPEG
      val/<class>/*.JPEG
    """
    def __init__(
        self,
        root: str,
        increment: int = 100,
        batch_size: int = 256,
        num_workers: int = 8,
        class_order: Optional[List[int]] = None,
        image_size: int = 224,
        seed: int = 0,
    ):
        super().__init__()
        self.root = root
        self.increment = int(increment)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.class_order = class_order
        self.image_size = int(image_size)
        self.seed = int(seed)

        self.train_tf, self.test_tf = _make_transforms(self.image_size)
        self.task_id = 0
        self.seen_classes: List[int] = []
        self.current_classes: List[int] = []
        self.num_classes_total = None  # determined at setup

        self.buffer: Optional[ExemplarBuffer] = None

    def prepare_data(self):
        # No automatic download
        pass

    def setup(self, stage: Optional[str] = None):
        train = dsets.ImageFolder(os.path.join(self.root, "train"), transform=self.train_tf)
        val = dsets.ImageFolder(os.path.join(self.root, "val"), transform=self.test_tf)
        self._full_train = train
        self._full_val = val
        n_classes = len(train.classes)
        self.num_classes_total = n_classes

        if self.class_order is None:
            rng = np.random.RandomState(self.seed)
            order = list(range(n_classes))
            rng.shuffle(order)
            self.class_order = order

    def num_tasks(self) -> int:
        return int(np.ceil(self.num_classes_total / self.increment))

    def set_task_id(self, task_id: int):
        self.task_id = int(task_id)
        start = task_id * self.increment
        end = min((task_id + 1) * self.increment, self.num_classes_total)
        self.current_classes = self.class_order[start:end]
        self.seen_classes = sorted(self.class_order[:end])

        self._dataset_train = self._filter_dataset(self._full_train, self.current_classes)
        self._dataset_val = self._filter_dataset(self._full_val, self.seen_classes)

    def _filter_dataset(self, dataset: dsets.ImageFolder, cls_ids: List[int]) -> Dataset:
        # dataset.targets exists for ImageFolder in torchvision
        idx = [i for i, y in enumerate(dataset.targets) if y in cls_ids]
        return Subset(dataset, idx)

    def train_dataloader(self):
        if self.buffer is not None and len(self.buffer) > 0:
            mix = ConcatDataset([self._dataset_train, self.buffer.make_dataset()])
        else:
            mix = self._dataset_train
        return DataLoader(mix, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self._dataset_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return self.val_dataloader()

    def build_class_loader(self, class_id: int, train: bool, batch_size: int, shuffle: bool, num_workers: int):
        base = self._full_train if train else self._full_val
        idx = [i for i, y in enumerate(base.targets) if y == class_id]
        subset = Subset(base, idx)
        return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
