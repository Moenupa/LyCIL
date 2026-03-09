import copy
from typing import TYPE_CHECKING, Optional

import torch
from datasets import Dataset, DatasetDict, concatenate_datasets
from torch.utils.data import DataLoader
from torch.nn import functional as F

from ..constants import _X_COLUMN_NAME
from .transform import apply_dataset_transform

if TYPE_CHECKING:
    from collections.abc import Callable


@torch.no_grad()
def compute_nme(
    dataloader: "DataLoader",
    feature_extractor: "Callable[[torch.Tensor], torch.Tensor]",
    device: "torch.device",
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Compute the normalized mean feature vector for a given dataloader.

    Args:
        dataloader (DataLoader):
            Dataloader, usually only contains samples in a single class.
        feature_extractor (Callable[[torch.Tensor], torch.Tensor]):
            Function to extract features.
        device (torch.device): device to use for computation.

    Raises:
        TypeError: if ``dataloader`` does not yield dict or Tensor

    Returns:
        tuple: 2 tensors
            - 1d tensor, sample-wise mean of feature vector, shaped (n_features),
            - 2d tensor, per-sample feature vectors, shaped (n_samples, n_features).

    """

    feature_list = []
    for batch in dataloader:
        if isinstance(batch, dict):
            x = batch[_X_COLUMN_NAME].to(device)
        elif isinstance(batch, torch.Tensor):
            x = batch.to(device)
        else:
            raise TypeError(f"batch must be dict or Tensor, got {type(batch)}")

        features = F.normalize(feature_extractor(x), dim=1).cpu()
        feature_list.append(features)

    # shaped (n_samples, n_features)
    per_sample_features = torch.cat(feature_list, dim=0)
    nme = per_sample_features.mean(dim=0)
    return nme, per_sample_features


class BaseExemplarBuffer(DatasetDict):
    r"""Fixed-size per-class exemplar buffer backed by a :class:`~datasets.DatasetDict`.

       Supports two memory management modes:

       - **Adaptive** (``mem_size`` given): total budget is fixed; per-class quota
         shrinks as new classes arrive via
         :math:`\lfloor mem\_size / n\_classes\_seen \rfloor`.
       - **Fixed** (``mem_size_per_class`` given): each class always stores the same
         number of exemplars regardless of the number of tasks seen.

       Exactly one of ``mem_size`` or ``mem_size_per_class`` must be provided.

       Args:
           mem_size (int | None, optional): Total exemplar budget across all classes.
               Mutually exclusive with ``mem_size_per_class``. (default: ``None``)
           mem_size_per_class (int | None, optional): Fixed per-class exemplar quota.
               Mutually exclusive with ``mem_size``. (default: ``None``)
           args: Positional arguments forwarded to :class:`~datasets.DatasetDict`.
           kwargs: Keyword arguments forwarded to :class:`~datasets.DatasetDict`.

       Raises:
           ValueError: If neither or both of ``mem_size`` / ``mem_size_per_class`` are
               provided, or if the supplied value is not a positive integer.
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

    def __init__(
        self,
        *args,
        mem_size: int | None = None,
        mem_size_per_class: int | None = None,
        **kwargs,
    ):
        if (mem_size is None) == (mem_size_per_class is None):
            raise ValueError(
                "Exactly one of `mem_size` or `mem_size_per_class` should be given. "
                + f"Got {mem_size} and {mem_size_per_class}."
            )

        val_to_check = mem_size or mem_size_per_class
        if not isinstance(val_to_check, int) or val_to_check <= 0:
            raise ValueError(
                "`mem_size` or `mem_size_per_class` must be positive integers."
            )

        super().__init__(*args, **kwargs)

        # exactly one of the two is None
        self.mem_size = mem_size
        self.mem_size_per_class = mem_size_per_class

        # two dicts keyed by class_id, manual sync required
        # per_class_data is `self` in DatasetDict format
        self.per_class_means: dict[int, torch.Tensor] = {}

    def __getitem__(self, k):
        if isinstance(k, int):
            raise KeyError("Avoid integer indexing. Use `str()` for classid keys.")

        return super().__getitem__(k)

    @property
    def mem_size_used(self) -> int:
        """Total number of exemplars currently stored."""
        return sum(len(subset) for subset in self.values())

    @property
    def is_adaptive(self) -> bool:
        """Whether to allow adaptive per-class size.
        If adaptive: per-class size is computed as `mem_size // num_classes`.
        E.g., if given total=200, per-class size is 20 at 10 classes, 10 at 20 classes, etc.

        Returns:
            bool: True if per-class size is adaptive.
        """
        return self.mem_size_per_class is None

    def size_per_class(self, target_num_classes: int) -> int:
        """Calculate per-class quota, if given a target number of seen classes.

        Args:
            target_num_classes (int):
                Target number of classes, to calculate the per-class quota for.

        Returns:
            int: Number of exemplars to keep per class for the target number of classes.
        """
        if self.mem_size_per_class is not None:
            return self.mem_size_per_class

        # otherwise, adaptive because mem_size must not be None
        # Example: if final target is 100 classes and we want 20 exemplars/class
        # at convergence, set mem_size=2000. Then task 0 with 20 seen classes gets
        # 2000 // 20 = 100 exemplars/class, and later tasks will shrink this quota
        # as more classes are introduced.
        assert self.mem_size is not None
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
        r"""Reduce exemplars, typically called after new classes arrive.

        Args:
            per_class_quota (int): Maximum number of exemplars to keep per class.
            trim_func (Callable[[Dataset, int], Dataset] | None, optional):
                Function to trim exemplars: ``trim_func(dataset, quota) -> trimmed_dataset``
                If None, get first `quota` samples. (default: None)

        """
        trim_func = trim_func or (
            lambda dataset, q: dataset if len(dataset) <= q else dataset.select(range(q))
        )

        for _class_id, _data in self.items():
            base = _data if isinstance(_data, Dataset) else Dataset.from_dict(_data)
            trimmed = trim_func(base, per_class_quota)

            if not isinstance(trimmed, Dataset):
                raise TypeError(
                    "trim_func must return a `datasets.Dataset` to preserve features."
                )

            trimmed.reset_format()
            self[_class_id] = trimmed

            if len(self[_class_id]) > per_class_quota:
                raise ValueError(
                    f"trim_func returned more than {per_class_quota} exemplars."
                )

    def make_dataset(
        self,
        keys: str | list[str] | None = None,
        transform: Callable | None = None,
    ) -> "Dataset":
        """Concatenate exemplar subsets into a single :class:`~datasets.Dataset`.

        Args:
            keys (str | list[str] | None, optional): Class-id string keys to
                include. If ``None``, all stored classes are included.
                (default: ``None``)
            transform (Callable | None, optional): Transform applied to samples.

        Returns:
            Dataset: Concatenated dataset from the requested class subsets.
        """
        if isinstance(keys, str):
            keys = [keys]

        subsets = []
        for k, v in self.items():
            if keys is None or k in keys:
                ds = v if isinstance(v, Dataset) else Dataset.from_dict(v)
                ds.reset_format()
                subsets.append(ds)
        ret = concatenate_datasets(subsets)

        apply_dataset_transform(ret, transform=transform)
        return ret

    def get_dataloader(
            self,
            keys: str | list[str] | None = None,
            transform: Callable | None = None,
            loader_kwargs: dict | None = None,
    ) -> DataLoader:
        """Build a :class:`~torch.utils.data.DataLoader` over exemplar subsets.

        Args:
            keys (str | list[str] | None, optional): Selected class keys. If
                ``None``, use all classes.
            transform (Callable | None, optional): Transform applied to samples.
            loader_kwargs (dict | None, optional): Extra arguments for
                :class:`~torch.utils.data.DataLoader`.

        Returns:
            DataLoader: Dataloader for the selected subsets.
        """
        dataset = self.make_dataset(keys=keys, transform=transform)
        return DataLoader(dataset, **(loader_kwargs or {}))
