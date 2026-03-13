import os.path as osp

import pytest

from lycil.data.hfmodule import HFDataModule


@pytest.fixture
def cifar10_path() -> str | None:
    path = "data/cifar10"
    if not osp.exists(path):
        return None

    return path


@pytest.mark.slow
def test_setup_w_seed(cifar10_path: str | None):
    if cifar10_path is None:
        pytest.skip("CIFAR10 dataset not found.")
        return

    import lightning as L

    L.seed_everything(42)
    cifar10_hfmodule = HFDataModule(
        path=cifar10_path,
        num_tasks=10,
    )

    cifar10_hfmodule.prepare_data()
    cifar10_hfmodule.setup(stage="fit")

    # expected class order:
    # [4, 5, 0, 3, 7, 2, 1, 9, 6, 8]

    train_loader = cifar10_hfmodule.train_dataloader()
    for batch in train_loader:
        assert "img" in batch and "label" in batch
        for label in batch["label"]:
            assert label.item() == 4

    cifar10_hfmodule.set_current_task(4)
    train_loader = cifar10_hfmodule.train_dataloader()
    for batch in train_loader:
        assert "img" in batch and "label" in batch
        for label in batch["label"]:
            assert label.item() == 7


@pytest.mark.slow
def test_setup_w_custom_labelset(cifar10_path: str | None):
    if cifar10_path is None:
        pytest.skip("CIFAR10 dataset not found.")
        return

    cifar10_hfmodule = HFDataModule(
        path=cifar10_path,
        label_map={i: i for i in range(10)},
        num_tasks=5,
    )
    cifar10_hfmodule.prepare_data()
    cifar10_hfmodule.setup(stage="fit")

    train_loader = cifar10_hfmodule.train_dataloader()
    for batch in train_loader:
        assert "img" in batch and "label" in batch
        for label in batch["label"]:
            assert label.item() in {0, 1}

    cifar10_hfmodule.set_current_task(3)
    train_loader = cifar10_hfmodule.train_dataloader()
    for batch in train_loader:
        assert "img" in batch and "label" in batch
        for label in batch["label"]:
            assert label.item() in {6, 7}

    test_loader = cifar10_hfmodule.test_dataloader()
    if isinstance(test_loader, list):
        test_loader = test_loader[-2]
    seen_labels = set()
    for batch in test_loader:
        assert "img" in batch and "label" in batch
        for label in batch["label"]:
            seen_labels.add(label.item())
    # we have seen task_id \in [0, 3] => 8 classes so far
    assert seen_labels == set(range(8))


@pytest.mark.slow
def test_buffer(cifar10_path: str | None):
    if cifar10_path is None:
        pytest.skip("CIFAR10 dataset not found.")
        return

    cifar10_hfmodule = HFDataModule(
        path=cifar10_path,
        num_tasks=10,
        label_map={i: i for i in range(10)},
        buffer_kwargs={"mem_size_per_class": 20},
    )
    cifar10_hfmodule.prepare_data()
    cifar10_hfmodule.setup(stage="fit")

    assert cifar10_hfmodule.buffer is not None

    cifar10_hfmodule.buffer["0"] = cifar10_hfmodule.get_filtered_dataset(
        split=cifar10_hfmodule._split_train,
        filter_fn=lambda e: e["label"] == 0,
        transform_name=cifar10_hfmodule.get_effective_transform_name("train"),
    )

    seen_labels = set()
    cifar10_hfmodule.set_current_task(2)
    train_loader = cifar10_hfmodule.train_dataloader()
    for batch in train_loader:
        assert "img" in batch and "label" in batch
        for label in batch["label"]:
            seen_labels.add(label.item())
    # we have seen label==2 training data, and label==0 from buffer replay
    assert seen_labels == {0, 2}


@pytest.mark.slow
@pytest.mark.parametrize(
    "task_id, expected_label_set, expected_num_old_classes, expected_num_seen_classes",
    [
        (0, {0, 1}, 0, 2),
        (1, {2, 3, 4}, 2, 5),
        (2, {5, 6, 7, 8, 9}, 5, 10),
    ],
)
def test_irregular_num_tasks(
    cifar10_path: str | None,
    task_id: int,
    expected_label_set: set,
    expected_num_old_classes: int,
    expected_num_seen_classes: int,
):
    if cifar10_path is None:
        pytest.skip("CIFAR10 dataset not found.")
        return

    cifar10_hfmodule = HFDataModule(
        path=cifar10_path,
        num_classes_per_task=[2, 3, 5],
        label_map={i: i for i in range(10)},
    )
    cifar10_hfmodule.prepare_data()
    cifar10_hfmodule.setup(stage="fit")

    seen_labels = set()
    cifar10_hfmodule.set_current_task(task_id)
    train_loader = cifar10_hfmodule.train_dataloader()
    for batch in train_loader:
        assert "img" in batch and "label" in batch
        for label in batch["label"]:
            seen_labels.add(label.item())
    # we have seen label==2 training data, and label==0 from buffer replay
    assert seen_labels == expected_label_set

    assert cifar10_hfmodule.num_tasks == 3
    assert cifar10_hfmodule.num_old_classes == expected_num_old_classes
    assert cifar10_hfmodule.num_seen_classes == expected_num_seen_classes
