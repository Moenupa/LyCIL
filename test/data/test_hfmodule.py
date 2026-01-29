import os.path as osp

import pytest

from lycil.data.hfmodule import HFDataModule


@pytest.fixture
def cifar10_hfmodule() -> HFDataModule | None:
    path = "data/cifar10"
    if not osp.exists(path):
        return None

    return HFDataModule(path=path)


def test_setup_w_seed(cifar10_hfmodule: HFDataModule | None):
    if cifar10_hfmodule is None:
        pytest.skip("CIFAR10 dataset not found.")
        return

    cifar10_hfmodule.seed = 42
    cifar10_hfmodule.num_tasks = 10

    cifar10_hfmodule.prepare_data()
    cifar10_hfmodule.setup(stage="fit")

    assert cifar10_hfmodule.labelset_per_task == {
        0: {4},
        1: {5},
        2: {0},
        3: {3},
        4: {7},
        5: {2},
        6: {1},
        7: {9},
        8: {6},
        9: {8},
    }

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

    cifar10_hfmodule.labelset_per_task = {
        i: {
            i,
        }
        for i in range(10)
    }

    cifar10_hfmodule.prepare_data()
    cifar10_hfmodule.setup(stage="fit")

    train_loader = cifar10_hfmodule.train_dataloader()
    for batch in train_loader:
        assert "img" in batch and "label" in batch
        for label in batch["label"]:
            assert label.item() == 0

    cifar10_hfmodule._cur_task_id = 7
    train_loader = cifar10_hfmodule.train_dataloader()
    for batch in train_loader:
        assert "img" in batch and "label" in batch
        for label in batch["label"]:
            assert label.item() == 7
