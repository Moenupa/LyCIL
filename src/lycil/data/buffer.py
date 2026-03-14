import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import torch
from datasets import Dataset, DatasetDict, concatenate_datasets
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ..constants import _X_COLUMN_NAME
from .util import deterministic_choice

if TYPE_CHECKING:
    from collections.abc import Callable


@torch.no_grad()
def compute_nme(
    dataloader: "DataLoader",
    feature_extractor: "Callable[[torch.Tensor], torch.Tensor]",
    device: "torch.device",
    normalize_mean: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Compute the (normalized) mean feature vector for a given dataloader.

    Args:
        dataloader (DataLoader):
            Dataloader, usually only contains samples in a single class.
        feature_extractor (Callable[[torch.Tensor], torch.Tensor]):
            Function to extract features.
        device (torch.device):
            device to use for computation.
        normalize_mean (bool, optional):
            Whether to L2-norm mean feature vector. (default: ``True``)

    Raises:
        TypeError: if ``dataloader`` does not yield dict or Tensor

    Returns:
        tuple: 2 tensors
            - 1d tensor, (normalized) all-sample feature mean, shaped (n_features),
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

        features = F.normalize(feature_extractor(x), dim=1)
        feature_list.append(features)

    # shaped (n_samples, n_features)
    per_sample_features = torch.cat(feature_list, dim=0)
    # shaped (n_features), mean across all samples in the dataloader
    mean_feature = per_sample_features.mean(dim=0)
    if normalize_mean:
        mean_feature = F.normalize(mean_feature, dim=0)
    return mean_feature, per_sample_features


@torch.no_grad()
def predict_nme_rank(
    feature_extractor: "Callable[[torch.Tensor], torch.Tensor]",
    x: torch.Tensor,
    class_ids: torch.Tensor,
    class_means: torch.Tensor,
    topk: int = 1,
) -> torch.Tensor:
    feats = F.normalize(feature_extractor(x), dim=1)
    means = F.normalize(class_means.to(feats.device), dim=1)
    class_ids = class_ids.to(feats.device)

    dists = torch.cdist(feats, means, p=2).pow(2)
    rank = torch.argsort(dists, dim=1)[:, :topk]
    return class_ids[rank]


@torch.no_grad()
def select_exemplar(
    size: int,
    dataset_size: int,
    dataloader: "DataLoader",
    feature_extractor: "Callable[[torch.Tensor], torch.Tensor]",
    device: "torch.device",
    strategy: Literal["random", "herding"] = "herding",
    seed_offset: int = 0,
) -> list[int]:
    if size <= 0 or size > dataset_size:
        raise ValueError(
            f"Cannot select exemplars of size {size} from dataset with {dataset_size} samples."
        )

    if strategy == "random":
        return deterministic_choice(0, dataset_size, size=size, seed_offset=seed_offset)

    # herding by default, edge cases should be handled by code linting
    class_mean, per_sample_features = compute_nme(dataloader, feature_extractor, device)

    selected_idx = []
    selected_mask = torch.zeros(
        dataset_size, dtype=torch.bool, device=per_sample_features.device
    )
    running_sum = torch.zeros_like(class_mean)

    # select topk exemplars iteratively,
    # each time picking the one that brings the mean closer to the class mean
    for k in range(size):
        candidate_idx = (~selected_mask).nonzero(as_tuple=False).squeeze(1)
        candidate_feats = per_sample_features[candidate_idx]

        mu_p = (running_sum.unsqueeze(0) + candidate_feats) / (k + 1)
        dist = torch.norm(class_mean.unsqueeze(0) - mu_p, p=2, dim=1)
        best_rel: int = torch.argmin(dist).item()  # ty: ignore[invalid-assignment]
        best_abs: int = candidate_idx[best_rel].item()  # ty: ignore[invalid-assignment]

        selected_idx.append(best_abs)
        selected_mask[best_abs] = True
        running_sum += per_sample_features[best_abs]

    return selected_idx


@dataclass
class BufferReplayArgs:
    strategy: Literal["random", "herding"] = field(
        default="herding",
        metadata={
            "help": "Exemplar selection strategy (default: 'herding'): "
            "1. 'herding' iteratively selects samples bringing closer to the class mean"
            "2. 'random' seeded by PL_GLOBAL_SEED + seed_offset for reproducibility. (default: 'herding')"
        },
    )
    loader_kwargs: dict = field(
        default_factory=lambda: {"batch_size": 128, "shuffle": False, "num_workers": 8},
        metadata={
            "help": "DataLoader keyword arguments during BufferReplay (Exemplar update)."
        },
    )
    eval: bool = field(
        default=True,
        metadata={
            "help": "Meta-Toggle to enable NME (nearest mean of exemplars) accuracy at the end of each task."
            " A common metric for exemplar-based CL methods to evaluate quality of exemplars."
        },
    )
    eval_every_n_epochs: int = field(
        default=20,
        metadata={"help": "Frequency of NME accuracy evaluation."},
    )
    eval_topk: int = field(
        default=1,
        metadata={"help": "Top-k accuracy of NME accuracy."},
    )
    eval_compute_cur_task: bool = field(
        default=True,
        metadata={"help": "Toggle to compute current task during NME evaluation."},
    )
    eval_recompute_old_task: bool = field(
        default=True,
        metadata={"help": "Toggle to recompute old tasks during NME evaluation."},
    )

    # validate eval_topk > 0
    def __post_init__(self):
        if self.eval_topk <= 0:
            raise ValueError(
                f"eval_topk must be a positive integer, got {self.eval_topk}."
            )
        if self.eval_every_n_epochs <= 0:
            raise ValueError(
                f"eval_every_n_epochs must be a positive integer, got {self.eval_every_n_epochs}."
            )


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
        mem_size (int | None, optional):
            Total exemplar budget across all classes.
            Mutually exclusive with ``mem_size_per_class``. (default: ``None``)
        mem_size_per_class (int | None, optional):
            Fixed per-class exemplar quota.
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
        r"""Create a BaseExemplarBuffer instance from an existing DatasetDict.

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
        r"""Whether the buffer uses an adaptive (total-budget) memory policy.

        - If ``True``, per-class quota should be dynamically computed as
        :math:`\lfloor\frac{\text{mem\_size}}{\text{n\_classes}}\rfloor`
        - If ``False``, each class always keeps exactly ``mem_size_per_class`` exemplars.
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

    @staticmethod
    def _trim_dataset(dataset: Dataset, quota: int) -> Dataset:
        if len(dataset) <= quota:
            return dataset

        return dataset.select(range(quota))

    def reduce_exemplars(
        self,
        per_class_quota: int,
        trim_func: "Callable[[Dataset, int], Dataset] | None" = None,
    ) -> None:
        r"""Reduce exemplars, typically called after new classes arrive.

        Args:
            per_class_quota (int): Maximum number of exemplars to keep per class.
            trim_func (Callable[[Dataset, int], Dataset] | None, optional):
                Function to trim exemplars: ``trim_func(dataset, quota) -> trimmed_dataset``
                If None, get first :math:`q` samples. (default: None)

        """
        trim_func = trim_func or self._trim_dataset

        for _class_id, _data in self.items():
            self[_class_id] = trim_func(_data, per_class_quota)
            if len(self[_class_id]) > per_class_quota:
                raise ValueError(
                    f"trim_func returned more than {per_class_quota} exemplars."
                )

    def make_dataset(
        self, keys: str | list[str] | None = None, transform_name: str | None = None
    ) -> "Dataset":
        """Concatenate exemplar subsets into a single :class:`~datasets.Dataset`.

        Avoid ``transform_name`` unless no datamodule object is available.
        Prefer ``transform_name`` in datamodule, which applies to both data & buffer.
        Refer to :class:`~lycil.data.hfmodule.HFDataModule` and its ``get_dataloader()``.

        Args:
            keys (str | list[str] | None, optional): Class-id string keys to
                include. If ``None``, all stored classes are included.
                (default: ``None``)
            transform_name (str | None, optional): HuggingFace formatter name
                to set on the returned dataset. (default: ``None``)

        Returns:
            Dataset: Concatenated dataset from the requested class subsets.
        """
        if isinstance(keys, str):
            keys = [keys]

        # collect subsets by filter and concatenate
        subsets: list[Dataset] = [
            v if isinstance(v, Dataset) else Dataset.from_dict(v)
            for k, v in self.items()
            if keys is None or k in keys
        ]
        ret = concatenate_datasets(subsets)

        # modifying ret will not affect self.items(),
        # because concatenate_datasets creates a new Dataset object.
        if transform_name is not None:
            ret.set_format(transform_name)
        return ret

    def get_dataloader(
        self,
        keys: str | list[str] | None = None,
        transform_name: str | None = None,
        loader_kwargs: dict | None = None,
    ) -> "DataLoader":
        """Build a :class:`~torch.utils.data.DataLoader` over exemplar subsets.

        Args:
            keys (str | list[str] | None, optional): Class-id string keys to
                include. If ``None``, all stored classes are used.
                (default: ``None``)
            transform_name (str | None, optional): HuggingFace formatter name
                to apply. (default: ``None``)
            loader_kwargs (dict | None, optional): Keyword arguments forwarded
                to :class:`~torch.utils.data.DataLoader`. (default: ``None``)

        Returns:
            DataLoader: DataLoader over the selected exemplar subsets.
        """
        dataset = self.make_dataset(keys=keys, transform_name=transform_name)
        return DataLoader(dataset, **(loader_kwargs or {}))  # ty: ignore[invalid-argument-type]
