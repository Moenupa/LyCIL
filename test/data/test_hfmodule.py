import os.path as osp

import pytest

from lycil.data.hfmodule import HFDataModule


@pytest.fixture
def cifar10_hfmodule() -> HFDataModule | None:
    path = "data/cifar10"
    if not osp.exists(path):
        return None

    return HFDataModule(path=path, buffer_kwargs={"mem_size": 200})


def test_setup_w_seed(cifar10_hfmodule: HFDataModule | None):
    if cifar10_hfmodule is None:
        pytest.skip("CIFAR10 dataset not found.")
        return

    cifar10_hfmodule.seed = 42
    cifar10_hfmodule.num_tasks = 10

    cifar10_hfmodule.prepare_data()
    cifar10_hfmodule.setup(stage="fit")

    # expected class order:
    # [4, 5, 0, 3, 7, 2, 1, 9, 6, 8]

    train_loader = cifar10_hfmodule.train_dataloader()
    for batch in train_loader:
        assert "img" in batch and "label" in batch
        for label in batch["label"]:
            assert label.item() == 4

    cifar10_hfmodule._cur_task_id = 4
    train_loader = cifar10_hfmodule.train_dataloader()
    for batch in train_loader:
        assert "img" in batch and "label" in batch
        for label in batch["label"]:
            assert label.item() == 7


def test_setup_w_custom_labelset(cifar10_hfmodule: HFDataModule | None):
    if cifar10_hfmodule is None:
        pytest.skip("CIFAR10 dataset not found.")
        return

    cifar10_hfmodule.label_map = {i: i for i in range(10)}
    cifar10_hfmodule.num_tasks = 5

    cifar10_hfmodule.prepare_data()
    cifar10_hfmodule.setup(stage="fit")

    train_loader = cifar10_hfmodule.train_dataloader()
    for batch in train_loader:
        assert "img" in batch and "label" in batch
        for label in batch["label"]:
            assert label.item() in {0, 1}

    cifar10_hfmodule._cur_task_id = 3
    train_loader = cifar10_hfmodule.train_dataloader()
    for batch in train_loader:
        assert "img" in batch and "label" in batch
        for label in batch["label"]:
            assert label.item() in {6, 7}

    test_loader = cifar10_hfmodule.test_dataloader()
    seen_labels = set()
    for batch in test_loader:
        assert "img" in batch and "label" in batch
        for label in batch["label"]:
            seen_labels.add(label.item())
    # we have seen task_id \in [0, 3] => 8 classes so far
    assert seen_labels == set(range(8))


def test_buffer(cifar10_hfmodule: HFDataModule | None):
    if cifar10_hfmodule is None:
        pytest.skip("CIFAR10 dataset not found.")
        return

    cifar10_hfmodule.label_map = {i: i for i in range(10)}
    cifar10_hfmodule.num_tasks = 10

    cifar10_hfmodule.prepare_data()
    cifar10_hfmodule.setup(stage="fit")

    assert cifar10_hfmodule.buffer is not None

    cifar10_hfmodule.buffer["0"] = cifar10_hfmodule.get_filtered_dataset(
        split=cifar10_hfmodule._split_train,
        filter_fn=lambda e: e["label"] == 0,
        transform_name=cifar10_hfmodule.get_effective_transform_name("train"),
    )

    seen_labels = set()
    cifar10_hfmodule._cur_task_id = 2
    train_loader = cifar10_hfmodule.train_dataloader()
    for batch in train_loader:
        assert "img" in batch and "label" in batch
        for label in batch["label"]:
            seen_labels.add(label.item())
    # we have seen label==2 training data, and label==0 from buffer replay
    assert seen_labels == {0, 2}
