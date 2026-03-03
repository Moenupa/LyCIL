from collections.abc import Callable
from typing import Any, Literal

import lightning as L
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from torch.utils.data import DataLoader

from ..constants import _CLTASK_COLUMN_NAME, _Y_COLUMN_NAME
from .buffer import BaseExemplarBuffer
from .transform import register_tf_as_formatter
from .util import (
    SplitMapping,
    check_bijection,
    deterministic_shuffle,
    get_num_classes_per_task,
    get_or_identity,
    reverse_mapping,
)


class HFDataModule(L.LightningDataModule):
    _FORMAT_FALLBACK = "torch"

    def __init__(
        self,
        path: str,
        dataset_kwargs: dict | None = None,
        num_tasks: int | None = None,
        num_classes_per_task: int | list[int] | None = None,
        label_column_name: str = "label",
        label_map: dict[int, Any] | list[Any] | None = None,
        transform_name: str | None = None,
        train_loader_kwargs: dict | None = None,
        val_loader_kwargs: dict | None = None,
        test_loader_kwargs: dict | None = None,
        split_map: dict | None = None,
        buffer_kwargs: dict | None = None,
    ):
        super().__init__()
        self.path = path
        self.load_kwargs: dict = dataset_kwargs or {}
        self.split_map: dict = split_map or SplitMapping()
        self.label_map: dict[int, Any] | list[Any] | None = label_map

        self.num_classes_per_task: list[int]
        self._num_tasks = num_tasks
        self._num_classes_per_task = num_classes_per_task
        self.label_column_name = label_column_name

        self.transform_name = transform_name
        if transform_name is not None:
            register_tf_as_formatter(transform_name)

        self.train_loader_kwargs = train_loader_kwargs or {}
        self.val_loader_kwargs = val_loader_kwargs or {}
        self.test_loader_kwargs = test_loader_kwargs or {}

        self.buffer: BaseExemplarBuffer | None = (
            BaseExemplarBuffer(**buffer_kwargs) if buffer_kwargs is not None else None
        )
        self.train_filter_fn: Callable[[dict], bool] | None = None
        self.use_buffer: bool = True

        self._cur_task_id: int = 0
        self.dataset: DatasetDict

        # --- cache for eval subsets (indices) ---
        self._idx_cache: dict[tuple[str, int, str], list[int]] = {}  # (split, j, mode)->indices
        self._val_loader_names: list[str] = []
        self._test_loader_names: list[str] = []

    @property
    def num_tasks(self) -> int:
        return len(self.num_classes_per_task)

    @property
    def num_old_classes(self) -> int:
        return sum(self.num_classes_per_task[: self._cur_task_id])

    @property
    def num_seen_classes(self) -> int:
        return sum(self.num_classes_per_task[: self._cur_task_id + 1])

    def prepare_data(self):
        load_dataset("cifar100", cache_dir=self.path)

    def setup(self, stage: str | None = None):
        self.dataset: DatasetDict = load_dataset(self.path, **self.load_kwargs)

        train_set: Dataset = self.dataset[self._split_train]
        unique_labels: list = train_set.unique(self.label_column_name)

        self.num_classes_per_task = get_num_classes_per_task(
            num_classes_per_task=self._num_classes_per_task,
            num_tasks=self._num_tasks,
            num_classes=len(unique_labels),
        )

        if self.label_map is not None:
            check_bijection(self.label_map, unique_labels)

        idx2label = self.label_map or deterministic_shuffle(unique_labels)
        label2idx = reverse_mapping(idx2label)

        idx2taskid: list[int] = []
        for i, n in enumerate(self.num_classes_per_task):
            idx2taskid.extend([i] * n)

        y_not_used = len(unique_labels) - len(idx2taskid)
        if y_not_used > 0:
            idx2taskid.extend([-1] * y_not_used)
        elif y_not_used < 0:
            raise ValueError("num_classes_per_task exceeds dataset classes.")

        def _map_fn(e: dict) -> dict:
            class_idx = label2idx[e[self.label_column_name]]
            return {
                _Y_COLUMN_NAME: class_idx,
                _CLTASK_COLUMN_NAME: idx2taskid[class_idx],
            }

        self.dataset = self.dataset.map(_map_fn)

        # dataset changed -> clear cache
        self._idx_cache.clear()

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

    def get_effective_transform_name(self, mode: Literal["train", "test"] = "train") -> str | None:
        if mode not in {"train", "test"}:
            raise ValueError(f"expect mode in 'train'/'test', got {mode}")
        if self.transform_name is not None:
            return f"{self.transform_name}_{mode}"
        return self._FORMAT_FALLBACK

    # ---------- train filter (kept as-is) ----------
    def is_label_in_cur_task(self, e: dict) -> bool:
        return e[_CLTASK_COLUMN_NAME] == self._cur_task_id

    def get_filtered_dataset(
        self,
        split: str,
        filter_fn: Callable[[dict], bool],
        transform_name: str | None = None,
        use_buffer: bool = False,
    ) -> Dataset:
        subset = self.dataset[split].filter(filter_fn)
        if transform_name is not None:
            subset.set_format(transform_name)
        if use_buffer and self.buffer is not None and len(self.buffer) > 0:
            subset = concatenate_datasets([subset, self.buffer.make_dataset(transform_name=transform_name)])
        return subset

    def get_dataloader(
        self,
        split: str,
        filter_fn: Callable[[dict], bool],
        transform_name: str | None,
        loader_kwargs: dict,
        use_buffer: bool = False,
    ) -> DataLoader:
        subset = self.get_filtered_dataset(split, filter_fn, transform_name, use_buffer)
        return DataLoader(subset, **loader_kwargs)

    def train_dataloader(self):
        return self.get_dataloader(
            split=self._split_train,
            filter_fn=self.train_filter_fn or self.is_label_in_cur_task,
            transform_name=self.get_effective_transform_name("train"),
            loader_kwargs=self.train_loader_kwargs,
            use_buffer=self.use_buffer,
        )

    # ---------- eval helpers (cached) ----------
    def _eval_indices(self, split: str, j: int, mode: Literal["cum", "inc"]) -> list[int]:
        key = (split, j, mode)
        if key in self._idx_cache:
            return self._idx_cache[key]

        dset = self.dataset[split]
        if mode == "cum":
            # 0..j
            idx = [k for k, t in enumerate(dset[_CLTASK_COLUMN_NAME]) if t <= j]
        else:
            # only j
            idx = [k for k, t in enumerate(dset[_CLTASK_COLUMN_NAME]) if t == j]

        self._idx_cache[key] = idx
        return idx

    def _eval_subset(self, split: str, j: int, mode: Literal["cum", "inc"], tfm: str | None) -> Dataset:
        idx = self._eval_indices(split, j, mode)
        subset = self.dataset[split].select(idx)
        if tfm is not None:
            subset.set_format(tfm)
        return subset

    def _make_eval_loaders(self, split: str, loader_kwargs: dict):
        tfm = self.get_effective_transform_name("test")
        loaders, names = [], []
        for j in range(self._cur_task_id + 1):
            loaders.append(DataLoader(self._eval_subset(split, j, "cum", tfm), **loader_kwargs))
            names.append(f"cum/task{j}")
            loaders.append(DataLoader(self._eval_subset(split, j, "inc", tfm), **loader_kwargs))
            names.append(f"inc/task{j}")
        return loaders, names

    def val_dataloader(self):
        loaders, names = self._make_eval_loaders(self._split_val, self.val_loader_kwargs)
        self._val_loader_names = names
        return loaders

    def test_dataloader(self):
        loaders, names = self._make_eval_loaders(self._split_test, self.test_loader_kwargs)
        self._test_loader_names = names
        return loaders