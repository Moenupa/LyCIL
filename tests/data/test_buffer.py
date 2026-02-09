import os

import pytest
from datasets import Dataset, load_dataset

from lycil.data.buffer import BaseExemplarBuffer


@pytest.fixture
def buffer100():
    buffer = BaseExemplarBuffer(mem_size=100)
    for i in range(10):
        data = Dataset.from_list([{"input": j, "label": i} for j in range(10)])
        buffer[f"{i}"] = data
    return buffer


def test_mem_size_used(buffer100: BaseExemplarBuffer):
    assert buffer100.mem_size_used == 100


@pytest.mark.parametrize(
    "n_class, expected_size",
    [
        (1, 100),
        (3, 33),
        (5, 20),
        (7, 14),
        (10, 10),
        (11, 9),
        (100, 1),
    ],
)
def test_size_per_class(
    buffer100: BaseExemplarBuffer, n_class: int, expected_size: int
):
    assert buffer100.size_per_class(n_class) == expected_size


@pytest.mark.parametrize(
    "n_class",
    [-1, 0, 200],
)
def test_size_per_class_fail(buffer100: BaseExemplarBuffer, n_class: int):
    # expect ValueError for invalid n_class
    with pytest.raises(ValueError):
        buffer100.size_per_class(n_class)


@pytest.mark.parametrize("quota", [5, 2])
def test_reduce_exemplars(buffer100: BaseExemplarBuffer, quota: int):
    for _class_id, _data in buffer100.items():
        assert len(_data) == 10
    buffer100.reduce_exemplars(per_class_quota=quota, trim_func=None)
    for _class_id, _data in buffer100.items():
        assert len(_data["input"]) == quota
        assert len(_data["label"]) == quota


def test_make_dataset(buffer100: BaseExemplarBuffer):
    combined_dataset = buffer100.make_dataset()
    assert isinstance(combined_dataset, Dataset)
    assert len(combined_dataset) == 100


@pytest.mark.slow
@pytest.mark.parametrize("dataset_name", ["cifar10", "cifar100"])
def test_with_cifar_datasets(dataset_name):
    dataset_path = f"data/{dataset_name}/"
    if not os.path.exists(dataset_path):
        pytest.skip(f"Dataset path {dataset_path} does not exist.")

    dataset = load_dataset(dataset_path, split="test")
    buffer = BaseExemplarBuffer(mem_size=200)
    assert buffer.mem_size == 200
    assert buffer.mem_size_used == 0

    label_name = "label" if "label" in dataset.column_names else "coarse_label"
    for i in range(10):
        class_data = dataset.filter(lambda example: example[label_name] == i)
        assert isinstance(class_data, Dataset)
        buffer[f"{i}"] = class_data

    buffer.reduce_exemplars(per_class_quota=10)
    for _class_id, _data in buffer.items():
        assert len(_data) <= 10

    assert buffer.mem_size_used <= 200

    combined_dataset = buffer.make_dataset()
    assert isinstance(combined_dataset, Dataset)
    assert len(combined_dataset) <= 200
    print(combined_dataset)
    print(combined_dataset[0])


if __name__ == "__main__":
    test_with_cifar_datasets("cifar10")
