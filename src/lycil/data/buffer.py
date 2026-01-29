import copy
from typing import TYPE_CHECKING, Optional

import torch
from datasets import Dataset, DatasetDict, concatenate_datasets

if TYPE_CHECKING:
    from collections.abc import Callable


class BaseExemplarBuffer(DatasetDict):
    r"""
    Fixed-size buffer with per-class lists.

    Args:
        mem_size (int, optional): Exemplar size in total, each class gets
            :math:`\lfloor(\frac{mem\_size}{n_{classes\_seen}})\rfloor`. (default: 2000)
        args: Additional args passed to ``datasets.DatasetDict``.
        kwargs: Additional kwargs passed to ``datasets.DatasetDict``.
    """

    @classmethod
    def from_dataset_dict(
        cls,
        dataset_dict: DatasetDict,
        mem_size: int = 2000,
    ) -> "BaseExemplarBuffer":
        r"""
        Create a BaseExemplarBuffer instance from an existing DatasetDict.

        Args:
            dataset_dict (DatasetDict): The source DatasetDict.
            mem_size (int): Exemplar size in total. (default: 2000)

        Returns:
            BaseExemplarBuffer: A new instance of BaseExemplarBuffer.
        """
        buffer = cls(mem_size=mem_size)
        for key, dataset in dataset_dict.items():
            buffer[key] = copy.deepcopy(dataset)
        return buffer

    def __init__(self, *args, **kwargs):
        mem_size = kwargs.pop("mem_size", 2000)
        if not isinstance(mem_size, int) or mem_size <= 0:
            raise ValueError("`mem_size` must be a positive integer.")

        super().__init__(*args, **kwargs)
        self.mem_size: int = mem_size

        # two dicts keyed by class_id, manual sync required
        # per_class_data is `self` in DatasetDict format
        self.per_class_means: dict[int, torch.Tensor] = {}

    @property
    def mem_size_used(self) -> int:
        return sum(len(subset) for subset in self.values())

    def size_per_class(self, target_num_classes: int) -> int:
        if not isinstance(target_num_classes, int) or target_num_classes <= 0:
            raise ValueError("`target_num_classes` must be a positive integer.")
        if target_num_classes > self.mem_size:
            raise ValueError("`target_num_classes` cannot be larger than `mem_size`.")

        return self.mem_size // target_num_classes

    def reduce_exemplars(
        self,
        per_class_quota: int,
        trim_func: Optional["Callable[[Dataset, int], Dataset]"] = None,
    ) -> None:
        r"""
        Reduce exemplars, typically called after new classes arrive.

        Args:
            per_class_quota (int): Maximum number of exemplars to keep per class.
            trim_func (Callable[[Dataset, int], Dataset] | None, optional):
                Function to trim exemplars: ``trim_func(dataset, quota) -> trimmed_dataset``
                If None, get first `quota` samples. (default: None)
        """
        for _class_id, _data in self.items():
            if trim_func is not None:
                self[_class_id] = trim_func(_data, per_class_quota)
                if len(self[_class_id]) > per_class_quota:
                    raise ValueError(
                        f"trim_func returned more than {per_class_quota} exemplars."
                    )
            else:
                # equivalent to trim_func = lambda dataset, q: dataset[:q]
                self[_class_id] = _data[:per_class_quota]  # type: ignore

    def make_dataset(self) -> "Dataset":
        # adaptive -> Dataset
        subsets = [
            subset if isinstance(subset, Dataset) else Dataset.from_dict(subset)
            for subset in self.values()
        ]
        return concatenate_datasets(subsets)
