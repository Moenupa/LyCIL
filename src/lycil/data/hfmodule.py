from collections.abc import Callable
from typing import Any, Literal

import lightning as L
from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import DataLoader

from ..constants import _CLTASK_COLUMN_NAME, _Y_COLUMN_NAME
from .buffer import BaseExemplarBuffer
from .transform import register_tf_as_formatter
from .util import SplitMapping, deterministic_shuffle, get_or_identity


class HFDataModule(L.LightningDataModule):
    _FORMAT_FALLBACK = "torch"

    def __init__(
        self,
        path: str,
        dataset_kwargs: dict | None = None,
        num_tasks: int = 1,
        num_classes_per_task: int | None = None,
        label_column_name: str = "label",
        label_map: dict[int, Any] | list[Any] | None = None,
        transform_name: str | None = None,
        # DataLoader kwargs
        train_loader_kwargs: dict | None = None,
        val_loader_kwargs: dict | None = None,
        test_loader_kwargs: dict | None = None,
        # train/val/test map in case some dataset uses different split names
        split_map: SplitMapping | None = None,
        buffer_kwargs: dict | None = None,
        seed: int | None = 42,
    ):
        super().__init__()

        # load_dataset(path, **self.load_kwargs)
        self.path = path
        self.load_kwargs: dict = dataset_kwargs or {}

        # key mapping for customization, e.g.,
        # {"train": "your_custom_split_for_train", ...}
        self.split_map: dict = split_map or SplitMapping()
        self.label_map = label_map
        self.num_tasks = num_tasks
        self.num_classes_per_task: int = num_classes_per_task
        self.label_column_name = label_column_name
        # custom stage; fallback to lazy init in self.setup()

        self.transform_name = transform_name
        if transform_name is not None:
            register_tf_as_formatter(transform_name)

        self.train_loader_kwargs = train_loader_kwargs or {}
        self.val_loader_kwargs = val_loader_kwargs or {}
        self.test_loader_kwargs = test_loader_kwargs or {}
        self.seed = seed

        self.buffer: BaseExemplarBuffer | None = (
            BaseExemplarBuffer(**buffer_kwargs) if buffer_kwargs is not None else None
        )

        self._cur_task_id: int = 0
        self.dataset: DatasetDict

    def prepare_data(self):
        load_dataset(self.path)

    def setup(self, stage: str | None = None):
        self.dataset: DatasetDict = load_dataset(self.path, **self.load_kwargs)

        # lazy init, this requires all of:
        # 1. train split exists
        # 2. label_column_name exists
        # 3. deterministic uniform sampling of labels
        train_set: Dataset = self.dataset[self._split_train]
        unique_labels: list = train_set.unique(self.label_column_name)
        self.num_classes_per_task: int = self.num_classes_per_task or (
            len(unique_labels) // self.num_tasks
        )

        # a bijection that 0-indexed cl_class_idx -> label
        if self.label_map is not None:
            import warnings

            if len(self.label_map) != len(unique_labels):
                warnings.warn(
                    "`label_map` does not cover all unique labels. "
                    + f"DataModule(label_map={self.label_map}). Actual Data has {unique_labels}."
                )

            # TODO: check for duplicates

        # idx, v -> v, idx for faster remapping
        idx2label = self.label_map or deterministic_shuffle(unique_labels, self.seed)
        reverse_lookup: dict[Any, int] = {}
        if isinstance(idx2label, (list, tuple)):
            reverse_lookup = {v: idx for idx, v in enumerate(idx2label)}
        elif isinstance(idx2label, dict):
            reverse_lookup = {v: idx for idx, v in idx2label.items()}
        else:
            raise TypeError("label_map must be a tuple, list or dict.")

        def _map_fn(e: dict) -> dict:
            mapped_class_idx = reverse_lookup[e[self.label_column_name]]
            task_id_belonged = mapped_class_idx // self.num_classes_per_task
            return {
                _Y_COLUMN_NAME: mapped_class_idx,
                _CLTASK_COLUMN_NAME: task_id_belonged,
            }

        self.dataset = self.dataset.map(_map_fn)

    def is_label_in_cur_task(self, e: dict) -> bool:
        return e[_CLTASK_COLUMN_NAME] == self._cur_task_id

    def is_label_in_seen_task(self, e: dict) -> bool:
        return e[_CLTASK_COLUMN_NAME] <= self._cur_task_id

    def set_current_task(self, task_id: int):
        self._cur_task_id = task_id

    def get_current_task(self) -> int:
        return self._cur_task_id

    @property
    def _split_train(self) -> str:
        return get_or_identity(self.split_map, "train")

    @property
    def _split_val(self) -> str:
        return get_or_identity(self.split_map, "val")

    @property
    def _split_test(self) -> str:
        return get_or_identity(self.split_map, "test")

    def get_effective_transform_name(
        self, mode: Literal["train", "test"] = "train"
    ) -> str | None:
        """Get transform settings's name in effect, under the given ``mode``.

        A fallback is ``self._FORMAT_FALLBACK`` which defaults to 'torch',
        which enables :class:``dataset.Dataset`` to return PyTorch tensors.

        Args:
            mode (Literal["train", "test"], optional):
                Use transform train or test. (default: "train")

        Raises:
            ValueError: If mode is not "train" or "test".

        Returns:
            str | None: The format name to be used, or None if not set.
        """

        if mode not in {"train", "test"}:
            raise ValueError(
                f"Format/Transform: expect mode in 'train' or 'test', got {mode}."
            )

        if self.transform_name is not None:
            return f"{self.transform_name}_{mode}"
        elif self._FORMAT_FALLBACK is not None:
            return self._FORMAT_FALLBACK

        return None

    def get_filtered_dataset(
        self,
        split: str,
        filter_fn: Callable[[dict], bool],
        transform_name: str | None = None,
    ) -> Dataset:
        subset = self.dataset[split].filter(filter_fn)
        if transform_name is not None:
            subset.set_format(format_name=transform_name)
        return subset

    def get_dataloader(
        self,
        split: str,
        filter_fn: Callable[[dict], bool],
        transform_name: str | None,
        loader_kwargs: dict,
    ) -> DataLoader:
        subset = self.get_filtered_dataset(
            split=split,
            filter_fn=filter_fn,
            transform_name=transform_name,
        )
        return DataLoader(subset, **loader_kwargs)  # type: ignore

    def train_dataloader(self):
        return self.get_dataloader(
            split=self._split_train,
            filter_fn=self.is_label_in_cur_task,
            transform_name=self.get_effective_transform_name("train"),
            loader_kwargs=self.train_loader_kwargs,
        )

    def val_dataloader(self):
        return self.get_dataloader(
            split=self._split_val,
            filter_fn=self.is_label_in_seen_task,
            transform_name=self.get_effective_transform_name("test"),
            loader_kwargs=self.val_loader_kwargs,
        )

    def test_dataloader(self):
        return self.get_dataloader(
            split=self._split_test,
            filter_fn=self.is_label_in_seen_task,
            transform_name=self.get_effective_transform_name("test"),
            loader_kwargs=self.test_loader_kwargs,
        )
