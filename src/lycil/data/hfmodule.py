import lightning as L
from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import DataLoader

from .transform import register_tf_as_formatter
from .util import SplitMapping, chunk, deterministic_shuffle, get_or_identity


class HFDataModule(L.LightningDataModule):
    _FORMAT_FALLBACK = "torch"

    def __init__(
        self,
        path: str,
        dataset_kwargs: dict | None = None,
        num_tasks: int = 1,
        label_column_name: str = "label",
        labelset_per_task: dict[int, set] | None = None,
        # transforms
        transform_name: str | None = None,
        # DataLoader kwargs
        train_loader_kwargs: dict | None = None,
        val_loader_kwargs: dict | None = None,
        test_loader_kwargs: dict | None = None,
        # train/val/test map in case some dataset uses different split names
        split_map: SplitMapping | None = None,
        seed: int | None = 42,
    ):
        super().__init__()

        # load_dataset(path, **self.load_kwargs)
        self.path = path
        self.load_kwargs: dict = dataset_kwargs or {}

        # key mapping for customization, e.g.,
        # {"train": "your_custom_split_for_train", ...}
        self.split_map: dict = split_map or SplitMapping()
        if labelset_per_task is not None and num_tasks > 1:
            print(
                "Warning: both `labelset_per_task` and `num_tasks` are provided. Using `labelset_per_task`."
            )
            self.num_tasks = len(labelset_per_task)
            # verify that keys are 0..num_tasks-1
            expected_keys = set(range(self.num_tasks))
            if set(labelset_per_task.keys()) != expected_keys:
                raise ValueError(
                    f"`labelset_per_task` keys must be 0..{self.num_tasks}, but got {set(labelset_per_task.keys())}."
                )
            self.labelset_per_task = labelset_per_task
        else:
            self.num_tasks = num_tasks
            self.labelset_per_task: dict[int, set] = {}
        self.label_column_name = label_column_name
        # custom stage; fallback to lazy init in self.setup()

        self.transform_name = transform_name
        if transform_name is not None:
            # dry run attempt
            register_tf_as_formatter(transform_name)

        self.train_loader_kwargs = train_loader_kwargs or {}
        self.val_loader_kwargs = val_loader_kwargs or {}
        self.test_loader_kwargs = test_loader_kwargs or {}
        self.seed = seed

        self._cur_task_id: int = 0
        self.dataset: DatasetDict

    def prepare_data(self):
        load_dataset(self.path)

    def setup(self, stage: str | None = None):
        self.dataset: DatasetDict = load_dataset(self.path, **self.load_kwargs)
        assert isinstance(self.dataset, DatasetDict)

        # lazy init, this requires all of:
        # 1. train split exists
        # 2. label_column_name exists
        # 3. deterministic uniform sampling of labels
        # class-balance is not considered here
        if not self.labelset_per_task:
            train_set: Dataset = self.dataset[self._split_train]
            unique_labels = train_set.unique(self.label_column_name)
            unique_labels = deterministic_shuffle(unique_labels, self.seed)
            self.labelset_per_task = {
                task_id: set(labelset)
                for task_id, labelset in enumerate(chunk(unique_labels, self.num_tasks))
            }

    def is_label_in_cur_task(self, e: dict) -> bool:
        return e[self.label_column_name] in self.labelset_per_task[self._cur_task_id]

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

    def train_dataloader(self):
        subset = self.dataset[self._split_train].filter(
            self.is_label_in_cur_task,
        )
        if self.transform_name is not None:
            subset.set_format(f"{self.transform_name}_train")
        elif self._FORMAT_FALLBACK is not None:
            subset.set_format(self._FORMAT_FALLBACK)
        return DataLoader(subset, **self.train_loader_kwargs)

    def val_dataloader(self):
        subset = self.dataset[self._split_val].filter(
            self.is_label_in_cur_task,
        )
        if self.transform_name is not None:
            subset.set_format(f"{self.transform_name}_test")
        elif self._FORMAT_FALLBACK is not None:
            subset.set_format(self._FORMAT_FALLBACK)
        return DataLoader(subset, **self.val_loader_kwargs)

    def test_dataloader(self):
        subset = self.dataset[self._split_test]
        if self.transform_name is not None:
            subset.set_format(f"{self.transform_name}_test")
        elif self._FORMAT_FALLBACK is not None:
            subset.set_format(self._FORMAT_FALLBACK)
        return DataLoader(subset, **self.test_loader_kwargs)  # type: ignore
