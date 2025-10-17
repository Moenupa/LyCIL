from math import ceil

import lightning.pytorch as pl
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from .buffer import ExemplarBuffer


class CILDataset(Dataset):
    """
    Dataset with Continual/Incremental Learning (CIL) features.

    Args:
        dataset (Dataset): Base dataset.
        buffer (ExemplarBuffer, optional): Buffer for replaying, etc. Default: None.
    """

    def __init__(
        self,
        dataset: Dataset,
        buffer: ExemplarBuffer = None,
    ) -> None:
        super().__init__()
        # dataset must be a subclass of Dataset
        assert isinstance(dataset, Dataset), (
            f"{self.__class__.__name__}: dataset must be a subclass of Dataset, got {type(dataset)}"
        )

        if buffer is not None and len(buffer) > 0:
            self.dataset = ConcatDataset([dataset, buffer.make_dataset()])
        else:
            self.dataset = dataset

        assert len(self.dataset) > 0, f"{self.__class__.__name__}: dataset is empty"

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple:
        x, y = self.dataset[index]
        return x, y


class BaseCILDataModule(pl.LightningDataModule):
    """
    API class for data modules with Continual/Incremental Learning (CIL) features.

    Args:
        root: root directory of the dataset.
        num_class_total: total number of classes in the dataset.
        num_class_per_task: number of classes per task.
        batch_size: batch size for data loaders. Default: 64.
        num_workers: number of workers for data loaders. Default: 4.
        target_transform: list of class indices in the order they will appear. Default: None (random).

    Attributes:
        buffer: exemplar buffer to store samples from previous tasks, can be set externally.
        task_id: current task ID, set by ``set_task()``
        classes_current: indices in ``class_order`` for the current task, set by ``set_task()``
        classes_seen: indices in ``class_order`` seen so far, set by ``set_task()``

    Examples::

        >>> class MyCILDataModule(BaseCILDataModule):
        ...     # implement methods
        ...     pass
    """

    # real classid -> fake index mapping, smaller index appear in earlier tasks
    target_transform: callable
    # buffer
    buffer: ExemplarBuffer | None

    # task id, set by `set_task`
    task_id: int
    # indices in class_order for the current task, set by `set_task`
    classes_current: list[int]
    # indices in class_order seen so far, set by `set_task`
    classes_seen: list[int]

    def __init__(
        self,
        root: str,
        num_class_total: int,
        num_class_per_task: int,
        batch_size: int = 64,
        num_workers: int = 4,
        target_transform: callable = None,
    ):
        super().__init__()
        # root dir of the dataset
        self.root = root

        # CIL params:
        # number of classes per task, i.e. increment
        self.num_class_per_task = num_class_per_task
        # total number of classes
        self.num_class_total = num_class_total

        # data loader params
        self.batch_size = batch_size
        self.num_workers = num_workers

        # target transforms
        self.target_transform = target_transform

    @property
    def num_tasks(self) -> int:
        return ceil((self.num_class_total / self.num_class_per_task))

    @staticmethod
    def dataset_by_target(
        dataset: Dataset,
        selected_targets: set,
    ) -> Dataset:
        """
        Returns a filtered dataset

        Args:
            dataset (Dataset): dataset to be filtered
            selected_targets (set): targets to keep

        Returns:
            Dataset: filtered dataset
        """
        return Subset(
            dataset, [i for i, (_, y) in enumerate(dataset) if y in selected_targets]
        )

    def set_task(self, task_id: int | str) -> None:
        """Set the current task ID. This will also update `classes_current` and `classes_seen`."""
        assert isinstance(task_id, int), (
            f"{self.__class__.__name__}: task_id must be int, got {type(task_id)}"
        )
        assert 0 <= task_id < self.num_tasks, (
            f"{self.__class__.__name__}: task_id must be in [0, {self.num_tasks - 1}], got {task_id}"
        )

        self.task_id = task_id

        _index_l = task_id * self.num_class_per_task
        _index_r = min(_index_l + self.num_class_per_task, self.num_class_total)

        self.classes_current = list(range(_index_l, _index_r))
        self.classes_seen = list(range(0, _index_r))

    def build_loader(self, dataset: Dataset, **kwargs) -> None:
        """
        Build a data loader for the given dataset.

        Args:
            dataset (Dataset): The dataset to load.
            kwargs: additional args for :ref:`DataLoader <torch.utils.data.DataLoader>`

        Returns:
            DataLoader: The data loader for the dataset.
        """
        kwargs.setdefault("batch_size", self.batch_size)
        kwargs.setdefault("num_workers", self.num_workers)
        return DataLoader(dataset, **kwargs)
