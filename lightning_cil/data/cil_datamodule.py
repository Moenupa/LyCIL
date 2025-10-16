from abc import ABC, abstractmethod


class AbstractCILDataModule(ABC):
    """
    API class for data modules with Continual/Incremental Learning (CIL) features.

    Attributes:
        num_class_total: total number of classes in the dataset.
        num_class_per_task: number of classes per task.
        classes_current: list of class IDs in the current task.
        classes_seen: list of all class IDs seen so far.
        class_order: optional list specifying the order of class IDs. If None, random shuffled by ``self.seed``.

    Examples::

        >>> class MyCILDataModule(LightningDataModule, AbstractCILDataModule):
        ...     # implement all abstract methods
        ...     pass
    """

    # total number of classes
    num_class_total: int
    # number of classes per task
    num_class_per_task: int

    # class-ids in the current task, set by `set_task`
    classes_current: list[int]
    # all class-ids seen so far, set by `set_task`
    classes_seen: list[int]

    # class-ids in the order they will appear
    class_order: list[int] | None

    @property
    @abstractmethod
    def num_tasks(self) -> int:
        """Number of tasks during CIL."""
        pass

    @abstractmethod
    def set_task(self, task_id: int | str) -> None:
        """Set the current task ID. This will also update `classes_current` and `classes_seen`."""
        pass
