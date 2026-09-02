from collections.abc import Callable
from functools import partial
from typing import Any, Literal

import lightning as L
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from torch.utils.data import DataLoader

from ..constants import (
    _CLTASK_COLUMN_NAME,
    _Y_COLUMN_NAME,
    TEST_LOADER_KWARGS,
    TRAIN_LOADER_KWARGS,
)
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


def preprocess_for_cl(
    sample: dict[str, Any],
    label_column_name: str,
    label2idx: dict[Any, int],
    idx2taskid: list[int] | dict[int, int],
) -> dict[str, int]:
    """Map a dataset sample's label to a CL class index and task ID.

    Designed for use with :meth:`datasets.Dataset.map`.

    Args:
        sample (dict[str, Any]): A single dataset sample.
        label_column_name (str): Column name containing the original label.
        label2idx (dict[Any, int]): Mapping from original label to CL class index.
        idx2taskid (list[int] | dict[int, int]): Mapping from CL class index to task ID.

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


def filter_by_classid(sample: dict[str, Any], _min: int, _max: int) -> bool:
    return _min <= sample[_Y_COLUMN_NAME] < _max


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
        *,
        dataset_kwargs: dict | None = None,
        num_tasks: int | None = None,
        num_classes_per_task: int | list[int] | None = None,
        label_column_name: str = "label",
        label_map: dict[int, Any] | list[Any] | None = None,
        transform_name: str | None = None,
        # DataLoader kwargs
        train_loader_kwargs: dict | None = TRAIN_LOADER_KWARGS,
        val_loader_kwargs: dict | None = TEST_LOADER_KWARGS,
        test_loader_kwargs: dict | None = TEST_LOADER_KWARGS,
        # train/val/test map in case some dataset uses different split names
        split_map: SplitMapping | None = None,
        buffer_kwargs: dict | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        # load_dataset(path, **self.load_kwargs)
        self.path = path
        self.load_kwargs: dict = dataset_kwargs or {}

        # key mapping for customization, e.g.,
        # {"train": "your_custom_split_for_train", ...}
        self.split_map: SplitMapping = split_map or SplitMapping()
        self.label_map: dict[int, Any] | list[Any] | None = label_map

        # task config, lazy init
        self.num_classes_per_task: list[int]
        self._num_tasks = num_tasks
        self._num_classes_per_task = num_classes_per_task

        self.label_column_name = label_column_name
        # custom stage; fallback to lazy init in self.setup()

        # Keep the external config as a name/alias, but resolve it to a callable
        # when building datasets/dataloaders.
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

        self._cur_task_id: int = 0
        self.dataset: DatasetDict

        # --- cache for eval subsets (indices) ---
        self._idx_cache: dict[
            tuple[str, int, str], list[int]
        ] = {}  # (split, j, mode)->indices
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
        load_dataset(self.path)

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
        return get_or_identity(self.split_map, "train")  # ty: ignore[invalid-argument-type]

    @property
    def _split_val(self) -> str:
        return get_or_identity(self.split_map, "val")  # ty: ignore[invalid-argument-type]

    @property
    def _split_test(self) -> str:
        return get_or_identity(self.split_map, "test")  # ty: ignore[invalid-argument-type]

    def get_effective_transform_name(
        self, mode: Literal["train", "test"] = "train"
    ) -> str | None:
        """Return the active HuggingFace formatter name for the given mode.

        Falls back to ``self._FORMAT_FALLBACK`` (default ``"torch"``) when no
        named transform set is configured, so that
        :class:`~datasets.Dataset` always returns PyTorch tensors.

        Args:
            mode (Literal["train", "test"], optional): Whether to use the
                training or test augmentation variant. (default: ``"train"``)

        Returns:
            str | None: Formatter name to pass to
                :meth:`~datasets.Dataset.set_format`, or ``None`` if no
                fallback is configured.

        Raises:
            ValueError: If ``mode`` is not ``"train"`` or ``"test"``.
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
        use_buffer: bool = False,
        buffer_current_class_only: bool = False,
    ) -> Dataset:
        """Filter a dataset split and optionally merge with the exemplar buffer.

        Args:
            split (str): Dataset split name (e.g., ``"train"``).
            filter_fn (Callable[[dict], bool]): Per-sample predicate; samples
                for which it returns ``False`` are excluded.
            transform_name (str | None, optional): HuggingFace formatter name
                to set on the returned dataset. (default: ``None``)
            use_buffer (bool, optional): If ``True`` and a non-empty buffer
                exists, its exemplars are concatenated to the filtered subset.
                (default: ``False``)
            buffer_current_class_only (bool, optional): If ``True``, only exemplars
                from the current class are used. (default: ``False``)

        Returns:
            Dataset: Filtered (and optionally buffer-augmented) dataset.
        """
        subset = self.dataset[split].filter(filter_fn)
        if use_buffer and self.buffer is not None and len(self.buffer) > 0:
            buffer_dset = self.buffer.make_dataset(
                keys=[
                    str(i) for i in range(self.num_old_classes, self.num_seen_classes)
                ]
                if buffer_current_class_only
                else None
            )
            subset = concatenate_datasets([subset, buffer_dset])

        if transform_name is not None:
            subset.set_format(transform_name)
        return subset

    def get_dataloader(
        self,
        split: str,
        filter_fn: Callable[[dict], bool],
        transform_name: str | None,
        loader_kwargs: dict,
        use_buffer: bool = False,
        buffer_current_class_only: bool = False,
    ) -> DataLoader:
        """Build a DataLoader from a filtered split with optional buffer mixing.

        Args:
            split (str): Dataset split name (e.g., ``"train"``).
            filter_fn (Callable[[dict], bool]): Per-sample filter predicate.
            transform_name (str | None): HuggingFace formatter name to apply.
            loader_kwargs (dict): Keyword arguments forwarded to
                :class:`~torch.utils.data.DataLoader`.
            use_buffer (bool, optional): If ``True`` and a non-empty buffer
                exists, its exemplars are appended to the split.
                (default: ``False``)
            buffer_current_class_only (bool, optional): If ``True``, only exemplars
                from the current class are used. (default: ``False``)

        Returns:
            DataLoader: DataLoader over the filtered split.
        """
        subset = self.get_filtered_dataset(
            split=split,
            filter_fn=filter_fn,
            transform_name=transform_name,
            use_buffer=use_buffer,
            buffer_current_class_only=buffer_current_class_only,
        )
        return DataLoader(subset, **loader_kwargs)  # ty: ignore[invalid-argument-type]

    def train_dataloader(self):
        return self.get_dataloader(
            split=self._split_train,
            filter_fn=self.train_filter_fn or self.is_label_in_cur_task,
            transform_name=self.get_effective_transform_name("train"),
            loader_kwargs=self.train_loader_kwargs,
            use_buffer=True,
        )

    def val_dataloader(self):
        loaders = self._make_eval_loaders(self._split_val, self.val_loader_kwargs)
        self._val_loader_names = list(loaders.keys())
        return list(loaders.values())

    def test_dataloader(self):
        loaders = self._make_eval_loaders(self._split_test, self.test_loader_kwargs)
        self._test_loader_names = list(loaders.keys())
        return list(loaders.values())

    def _eval_indices(
        self, split: str, j: int, mode: Literal["cum", "inc"]
    ) -> list[int]:
        # TODO: this is rubbish code, but we will leave it until finalizing v1
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
        transform_name: str | None,
    ) -> Dataset:
        idx = self._eval_indices(split, j, mode)
        subset = self.dataset[split].select(idx)
        if transform_name is not None:
            subset.set_format(transform_name)
        return subset

    def _make_eval_loaders(
        self, split: str, loader_kwargs: dict
    ) -> dict[str, DataLoader]:
        transform_name = self.get_effective_transform_name("test")
        loaders: dict[str, DataLoader] = {}
        for j in range(self._cur_task_id + 1):
            loaders[f"cum/task{j}"] = DataLoader(
                self._eval_subset(split, j, "cum", transform_name),  # ty: ignore[invalid-argument-type]
                **loader_kwargs,
            )
            loaders[f"inc/task{j}"] = DataLoader(
                self._eval_subset(split, j, "inc", transform_name),  # ty: ignore[invalid-argument-type]
                **loader_kwargs,
            )
        return loaders
