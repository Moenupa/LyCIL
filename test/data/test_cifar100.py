import pytest

from lightning_cil.data.modules.cifar100 import CIFAR100DataModule


@pytest.mark.parametrize("num_class_per_task", [1, 2, 10])
def test_cifar100_datamodule(num_class_per_task: int):
    dm = CIFAR100DataModule(
        root="/public/datasets/cifar",
        num_class_per_task=num_class_per_task,
    )
    dm.prepare_data()
    dm.setup()

    assert dm.num_class_total == 100
    assert dm.num_class_per_task == num_class_per_task
    assert dm.num_tasks == 100 // num_class_per_task


@pytest.mark.parametrize(
    "task_id, train",
    [
        (0, True),
        (3, False),
        (5, False),
        (9, True),
    ],
)
def test_cifar100_datamodule_task(task_id: int, train: bool):
    dm = CIFAR100DataModule(
        root="/public/datasets/cifar",
        num_class_per_task=10,
    )
    dm.prepare_data()
    dm.setup()
    dm.set_task(task_id)

    # 10 classes, 0-9, 10-19, ..., 90-99
    expected_classes_current = set(range(task_id * 10, (task_id + 1) * 10))
    # 10*(task_id+1) seen classes, 0-9, 0-19, ..., 0-99
    expected_classes_seen = set(range(0, (task_id + 1) * 10))

    assert set(dm.classes_current) == expected_classes_current
    assert set(dm.classes_seen) == expected_classes_seen

    total = 0
    if train:
        for _, y in dm.train_dataloader():
            for yi in y:
                assert yi.item() in expected_classes_current, (
                    f"{yi.item()} not in {expected_classes_current}"
                )
                total += 1
        assert total == 5000
    else:
        for _, y in dm.val_dataloader():
            for yi in y:
                assert yi.item() in expected_classes_seen
                total += 1
        assert total == 1000 * (task_id + 1)
