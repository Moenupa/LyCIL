from math import ceil

import lightning.pytorch as pl

from .buffer import ExemplarBuffer


class BaseCILDataModule(pl.LightningDataModule):
    """
    API class for data modules with Continual/Incremental Learning (CIL) features.

    Attributes:
        root: root directory of the dataset.
        num_class_total: total number of classes in the dataset.
        num_class_per_task: number of classes per task.
        class_order: optional list specifying the order of class IDs. If None, random shuffled by ``self.seed``.
        buffer: exemplar buffer to store samples from previous tasks, can be set externally.
        classes_current: indices in ``class_order`` for the current task, set by ``set_task()``
        classes_seen: indices in ``class_order`` seen so far, set by ``set_task()``
        task_id: current task ID, set by ``set_task()``

    Examples::

        >>> class MyCILDataModule(BaseCILDataModule):
        ...     # implement methods
        ...     pass
    """

    # root dir of dataset
    root: str

    # total number of classes
    num_class_total: int
    # number of classes per task
    num_class_per_task: int

    # class-ids in the order they will appear
    class_order: list[int | str] | None
    # buffer
    buffer: ExemplarBuffer | None

    # task id, set by `set_task`
    task_id: int
    # indices in class_order for the current task, set by `set_task`
    classes_current: list[int]
    # indices in class_order seen so far, set by `set_task`
    classes_seen: list[int]

    @property
    def num_tasks(self) -> int:
        return ceil((self.num_class_total / self.num_class_per_task))

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
