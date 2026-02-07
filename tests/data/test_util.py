import lightning as L
import pytest

from lycil.data.util import deterministic_shuffle, get_num_classes_per_task


@pytest.mark.parametrize(
    "seed,before,after",
    [
        (42, list(range(10)), [5, 6, 0, 7, 3, 2, 4, 9, 1, 8]),
        (0, [0, 1, 2, 3], [2, 0, 1, 3]),
    ],
)
def test_determistics_shuffle(seed: int, before: list[int], after: list[int]):
    L.seed_everything(seed)
    actual = deterministic_shuffle(before)
    assert actual == after


@pytest.mark.parametrize(
    "num_classes_per_task,num_classes,num_tasks,expected",
    [
        (None, 10, None, ValueError),
        (1, 10, 11, ValueError),
        (1, 10, 10, [1] * 10),
        (None, 10, 3, [3] * 3),
        (4, 10, None, [4] * 2),
    ],
)
def test_get_num_classes_per_task(
    num_classes_per_task: int | list[int] | None,
    num_classes: int,
    num_tasks: int | None,
    expected: list[int] | type[Exception],
):
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            get_num_classes_per_task(num_classes_per_task, num_classes, num_tasks)
    else:
        actual = get_num_classes_per_task(num_classes_per_task, num_classes, num_tasks)
        assert actual == expected
