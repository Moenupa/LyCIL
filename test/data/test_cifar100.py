import pytest
from math import ceil

from lycil.data.modules.cifar100 import CIFAR100DataModule


@pytest.mark.parametrize("num_class_per_task", [4, 10, 30])
def test_cifar100_datamodule(num_class_per_task: int):
    dm = CIFAR100DataModule(
        root="data/cifar",
        num_class_per_task=num_class_per_task,
    )
    dm.prepare_data()
    dm.setup()

    # cifar100 should default to total of 100 classes
    assert dm.num_class_total == 100

    # and num_tasks * num_class_per_task ~ 100
    assert dm.num_class_per_task == num_class_per_task
    # round up because 100 = sum(30, 30, 30, 10) => 4 tasks
    assert dm.num_tasks == ceil(100 / num_class_per_task)


@pytest.mark.parametrize(
    "task_id, train, num_class_per_task",
    [
        (0, True, 20),
        (3, False, 30),
        (5, False, 10),
        (9, True, 10),
    ],
)
def test_cifar100_datamodule_task(task_id: int, train: bool, num_class_per_task: int):
    dm = CIFAR100DataModule(
        root="data/cifar",
        num_class_per_task=num_class_per_task,
        batch_size=1,
    )
    dm.prepare_data()
    dm.setup()
    dm.set_task(task_id)

    n_tasks_so_far = task_id + 1

    # task_id starts from 0
    upper_bound = min(100, n_tasks_so_far * num_class_per_task)

    # 10 classes, 0-9, 10-19, ..., 90-99
    expected_classes_current = set(range(task_id * num_class_per_task, upper_bound))
    assert set(dm.classes_current) == expected_classes_current

    # 10*(task_id+1) seen classes, 0-9, 0-19, ..., 0-99
    expected_classes_seen = set(range(0, upper_bound))
    assert set(dm.classes_seen) == expected_classes_seen

    # cifar100: 100 classes; per-class: 500 train, 100 test
    total = 0
    if train:
        for _, y in dm.train_dataloader():
            assert y.item() in expected_classes_current, (
                f"{y.item()} not in {expected_classes_current}"
            )
            total += 1
        # train contains current tasks, so 500 * current_classes
        assert total == 500 * len(expected_classes_current)
    else:
        val_loaders = dm.val_dataloader()

        # should be a list of dataloaders, one for each seen task
        assert len(val_loaders) == n_tasks_so_far

        for per_task_dataloader in val_loaders:
            for _, y in per_task_dataloader:
                assert y.item() in expected_classes_seen
                total += 1
        # val contains all seen tasks, so 100 * seen_classes
        assert total == 100 * len(expected_classes_seen)
