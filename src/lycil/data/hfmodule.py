from collections.abc import Callable
from functools import partial
from typing import Any, Literal

import lightning as L
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from torch.utils.data import DataLoader

from ..constants import _CLTASK_COLUMN_NAME, _Y_COLUMN_NAME
from .buffer import BaseExemplarBuffer
from .transform import apply_dataset_transform, get_transform
from .util import (
    SplitMapping,
    check_bijection,
    deterministic_shuffle,
    get_num_classes_per_task,
    get_or_identity,
    reverse_mapping,
)


def preprocess_for_cl(
        sample: dict[str, Any],
        label_column_name: str,
        label2idx: dict[Any, int],
        idx2taskid: dict[int, int],
) -> dict[str, int]:
    """Map a dataset sample's label to a CL class index and task ID.

    Designed for use with :meth:`datasets.Dataset.map`.

    Args:
        sample (dict[str, Any]): A single dataset sample.
        label_column_name (str): Column name containing the original label.
        label2idx (dict[Any, int]): Mapping from original label to CL class index.
        idx2taskid (dict[int, int]): Mapping from CL class index to task ID.

    Returns:
        dict[str, int]: Dict containing ``_Y_COLUMN_NAME`` (class index) and
            ``_CLTASK_COLUMN_NAME`` (task ID) entries.
    """
    class_idx = label2idx[sample[label_column_name]]
    task_id_belonged = idx2taskid[class_idx]
    return {
        _Y_COLUMN_NAME: class_idx,
        _CLTASK_COLUMN_NAME: task_id_belonged,
    }


def filter_by_task(sample: dict[str, Any], task_id: int) -> bool:
    return sample[_CLTASK_COLUMN_NAME] == task_id


def filter_by_classid(sample: dict[str, Any], class_idx: int) -> bool:
    return sample[_Y_COLUMN_NAME] == class_idx


class HFDataModule(L.LightningDataModule):
    """HuggingFace Datasets-backed data module for class-incremental learning.

    Loads a dataset via :func:`~datasets.load_dataset`, partitions classes into
    sequential CL tasks, remaps labels to contiguous zero-based indices, and
    provides per-task dataloaders with optional exemplar buffer integration.

    Args:
        path (str): Dataset identifier passed to :func:`~datasets.load_dataset`.
        dataset_kwargs (dict | None, optional): Extra keyword arguments forwarded
            to :func:`~datasets.load_dataset`. (default: ``None``)
        num_tasks (int | None, optional): Total number of CL tasks. Required when
            ``num_classes_per_task`` is ``None``. (default: ``None``)
        num_classes_per_task (int | list[int] | None, optional): Number of classes
            per task as a scalar or per-task list. (default: ``None``)
        label_column_name (str, optional): Column name for class labels in the raw
            dataset. (default: ``"label"``)
        label_map (dict[int, Any] | list[Any] | None, optional): Explicit mapping
            from CL class index to original label. If ``None``, labels are shuffled
            deterministically using the global seed. (default: ``None``)
        transform_name (str | None, optional): Name of a registered image transform
            set (e.g., ``"cifar10"``). (default: ``None``)
        train_loader_kwargs (dict | None, optional): Keyword arguments for the
            training :class:`~torch.utils.data.DataLoader`. (default: ``None``)
        val_loader_kwargs (dict | None, optional): Keyword arguments for the
            validation :class:`~torch.utils.data.DataLoader`. (default: ``None``)
        test_loader_kwargs (dict | None, optional): Keyword arguments for the
            test :class:`~torch.utils.data.DataLoader`. (default: ``None``)
        split_map (SplitMapping | None, optional): Custom split-name mapping for
            datasets that use non-standard split names. (default: ``None``)
        buffer_kwargs (dict | None, optional): Keyword arguments forwarded to
            :class:`~lycil.data.buffer.BaseExemplarBuffer`. If ``None``, no buffer
            is created. (default: ``None``)
    """
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

        # Keep the external config as a name/alias, but resolve it to a callable
        # when building datasets/dataloaders.
        self.transform_name = transform_name

        self.train_loader_kwargs = train_loader_kwargs or {}
        self.val_loader_kwargs = val_loader_kwargs or {}
        self.test_loader_kwargs = test_loader_kwargs or {}

        self.buffer: BaseExemplarBuffer | None = (
            BaseExemplarBuffer(**buffer_kwargs) if buffer_kwargs is not None else None
        )
        self.train_filter_fn: Callable[[dict], bool] | None = None

        self._cur_task_id: int = 0
        self.dataset: DatasetDict

        # --- cache for eval subsets (indices) ---
        self._idx_cache: dict[tuple[str, int, str], list[int]] = {}  # (split, j, mode)->indices
        self._val_loader_names: list[str] = []
        self._test_loader_names: list[str] = []



    @property
    def num_tasks(self) -> int:
        """Total number of CL tasks derived from ``num_classes_per_task``."""
        return len(self.num_classes_per_task)

    @property
    def num_old_classes(self) -> int:
        """Cumulative number of classes, introduced < current task."""
        return sum(self.num_classes_per_task[: self._cur_task_id])

    @property
    def num_seen_classes(self) -> int:
        """Cumulative number of classes, introduced <= current task."""
        return sum(self.num_classes_per_task[: self._cur_task_id + 1])

    def prepare_data(self):
        #TODO: fixbug
        # load_dataset(self.path)
        # ./data/cifar100
        load_dataset("cifar100", cache_dir=self.path)

    def setup(self, stage: str | None = None):
        self.dataset: DatasetDict = load_dataset(self.path, **self.load_kwargs)

        # lazy init, this requires all of:
        # 1. train split exists
        # 2. label_column_name exists
        # 3. deterministic uniform sampling of labels
        train_set: Dataset = self.dataset[self._split_train]
        unique_labels: list = train_set.unique(self.label_column_name)

        self.num_classes_per_task = get_num_classes_per_task(
            num_classes_per_task=self._num_classes_per_task,
            num_tasks=self._num_tasks,
            num_classes=len(unique_labels),
        )

        # a bijection that 0-indexed cl_class_idx -> label
        if self.label_map is not None:
            check_bijection(self.label_map, unique_labels)

        # idx, v -> v, idx for faster remapping
        idx2label = self.label_map or deterministic_shuffle(unique_labels)
        label2idx = reverse_mapping(idx2label)

        # construct mapping y -> cl_task_id
        idx2taskid: list[int] = []
        for i, num_classes in enumerate(self.num_classes_per_task):
            idx2taskid.extend([i] * num_classes)

        y_not_used = len(unique_labels) - len(idx2taskid)
        if y_not_used > 0:
            # pad with -1 for unused labels
            idx2taskid.extend([-1] * y_not_used)
        elif y_not_used < 0:
            raise ValueError(
                "The provided num_classes_per_task results in more classes than "
                "the actual number of unique labels in the dataset."
            )

        self.dataset = self.dataset.map(
            partial(
                preprocess_for_cl,
                label_column_name=self.label_column_name,
                label2idx=label2idx,
                idx2taskid=idx2taskid,
            )
        )

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

    def get_effective_transform(
            self, mode: Literal["train", "test"] = "train"
    ) -> Callable | None:
        """Return the active transform for the given mode.

        Args:
            mode (Literal["train", "test"], optional): Transform mode to use.
                Defaults to ``"train"``.

        Returns:
            Callable | None: The transform for the given mode, or ``None`` if no
            transform is configured.

        Raises:
            ValueError: If ``mode`` is not ``"train"`` or ``"test"``.
        """
        if mode not in {"train", "test"}:
            raise ValueError(f"expect mode in 'train'/'test', got {mode}")
        if self.transform_name is None:
            return None
        return get_transform(self.transform_name, mode)

    # ---------- train filter (kept as-is) ----------

    def get_filtered_dataset(
            self,
            split: str,
            filter_fn: Callable[[dict], bool],
            transform: Callable | None = None,
            use_buffer: bool = True,
            buffer_only_new: bool = False,
    ) -> Dataset:
        subset = self.dataset[split].filter(filter_fn)
        subset.reset_format()
        if use_buffer and self.buffer is not None and len(self.buffer) > 0:
            if buffer_only_new:
                buffer_set = self.buffer.make_dataset(
                    keys=[str(i) for i in range(self.num_old_classes, self.num_seen_classes)]
                )
            else:
                buffer_set = self.buffer.make_dataset()
            buffer_set.reset_format()
            subset = concatenate_datasets([subset, buffer_set])
        apply_dataset_transform(subset, transform=transform)
        return subset

    def get_dataloader(
            self,
            split: str,
            filter_fn: Callable[[dict], bool],
            transform: Callable | None,
            loader_kwargs: dict,
            use_buffer: bool = True,
            buffer_only_new: bool = False,
    ) -> DataLoader:
        """Build a dataloader from a filtered split with optional buffer mixing.

        Args:
            split (str): Dataset split name.
            filter_fn (Callable[[dict], bool]): Filter applied to each sample.
            transform (Callable | None): Transform applied to samples.
            loader_kwargs (dict): Extra arguments for
                :class:`~torch.utils.data.DataLoader`.
            use_buffer (bool, optional): Whether to append buffer exemplars.
                Defaults to ``True``.
            buffer_only_new (bool, optional): Whether to use only new exemplars
                from the buffer when ``use_buffer`` is enabled. Defaults to
                ``False``.

        Returns:
            DataLoader: Dataloader over the selected data.
        """
        subset = self.get_filtered_dataset(
            split=split,
            filter_fn=filter_fn,
            transform=transform,
            use_buffer=use_buffer,
            buffer_only_new=buffer_only_new,
        )
        return DataLoader(subset, **loader_kwargs)

    def train_dataloader(self):
        return self.get_dataloader(
            split=self._split_train,
            filter_fn=self.train_filter_fn or self.is_label_in_cur_task,
            transform=self.get_effective_transform("train"),
            loader_kwargs=self.train_loader_kwargs,
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

    def _eval_subset(
            self,
            split: str,
            j: int,
            mode: Literal["cum", "inc"],
            transform: Callable | None,
    ) -> Dataset:
        idx = self._eval_indices(split, j, mode)
        subset = self.dataset[split].select(idx)
        apply_dataset_transform(subset, transform=transform)
        return subset

    def _make_eval_loaders(self, split: str, loader_kwargs: dict):
        transform = self.get_effective_transform("test")
        loaders, names = [], []
        for j in range(self._cur_task_id + 1):
            loaders.append(
                DataLoader(self._eval_subset(split, j, "cum", transform), **loader_kwargs)
            )
            names.append(f"cum/task{j}")
            loaders.append(
                DataLoader(self._eval_subset(split, j, "inc", transform), **loader_kwargs)
            )
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
