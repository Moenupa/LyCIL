import pytest

from data.modules.cifar100 import CIFAR100DataModule


def test_cifar100_datamodule():
    dm = CIFAR100DataModule(
        root="/public/datasets/cifar",
        num_class_per_task=10,
    )
    dm.prepare_data()
    dm.setup()

    assert dm.num_class_total == 100
    assert dm.num_class_per_task == 10
    assert dm.num_tasks == 10


@pytest.mark.parametrize("task_id", [0, 5, 9])
def test_cifar100_datamodule_task(task_id: int):
    dm = CIFAR100DataModule(
        root="/public/datasets/cifar",
        num_class_per_task=10,
    )
    dm.prepare_data()
    dm.setup()

    dm.set_task(task_id)

    assert len(dm.classes_current) == 10
    assert len(dm.classes_seen) == (task_id + 1) * 10

    train_dataset = dm._dataset_train
    test_dataset = dm._dataset_test
    assert len(train_dataset) == 5000
    assert len(test_dataset) == 1000 * (task_id + 1)
