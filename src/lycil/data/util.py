from typing import Any, TypedDict

from ..constants import get_seed


def get_or_identity(mapping: dict[str, str], key: str):
    """Return mapped value for key, or key itself when absent."""
    # mapping = mapping or IDENTITY_MAPPING

    # custom key_name support with fallback `key_name`
    # 1. split names: train/val/test
    # 2. column names: input/output/image/label
    # for mapping `self.splitmapping`, key is standardized, value is customized
    return mapping.get(key, key)


class SplitMapping(TypedDict, total=False):
    """Optional mapping from standard split names to dataset-specific names.

    All fields are optional. When a key is absent the standard name
    (``"train"``, ``"val"``, or ``"test"``) is used as-is.

    Attributes:
        train (str): Custom split name for training data.
        val (str): Custom split name for validation data.
        test (str): Custom split name for test data.
    """

    train: str
    val: str
    test: str


def deterministic_shuffle(lst: list) -> list:
    """Shuffle a list in place using the configured global seed.

    Args:
        lst (list): List to shuffle.

    Returns:
        list: The shuffled list (same object as input).
    """
    from numpy.random import default_rng

    default_rng(get_seed()).shuffle(lst)
    return lst


def get_num_classes_per_task(
    num_classes_per_task: int | list[int] | None,
    num_classes: int,
    num_tasks: int | None,
) -> list[int]:
    """Resolve a per-task class-count schedule from user-provided settings.

    Args:
        num_classes_per_task (int | list[int] | None):
            Number of classes per task, either as int or list of ints for each task.
            If None, will be inferred from ``num_tasks``.
        num_classes (int):
            Total number of classes across all tasks.
        num_tasks (int | None):
            Total number of tasks. Required if ``num_classes_per_task`` is None.

    Returns:
        list[int]: A list of class counts per task.

    Raises:
        ValueError: If the provided settings are inconsistent or invalid.
    """
    # we want to find `N=n*tasks`, so at least one of them must be given
    if num_classes_per_task is None and num_tasks is None:
        raise ValueError("`num_classes_per_task` and `num_tasks` cannot be both None.")

    if isinstance(num_classes_per_task, list):
        if num_tasks is not None and len(num_classes_per_task) != num_tasks:
            raise ValueError(
                f"Length of num_classes_per_task {len(num_classes_per_task)} != num_tasks {num_tasks}."
            )
        if sum(num_classes_per_task) > num_classes:
            raise ValueError(
                f"Exceeding total classes: sum({num_classes_per_task}) > {num_classes}."
            )

        return num_classes_per_task

    # if num_classes_per_task is given as int,
    # return [num_classes_per_task, ...], if length not given, try max it
    if isinstance(num_classes_per_task, int):
        if num_tasks is not None and num_classes_per_task * num_tasks > num_classes:
            raise ValueError(
                f"Exceeding total classes: {num_classes_per_task} * {num_tasks} > {num_classes}."
            )

        num_tasks = num_tasks or (num_classes // num_classes_per_task)
        return [num_classes_per_task for _ in range(num_tasks)]

    # now num_classes_per_task is None, num_tasks must be given
    assert num_tasks is not None, (
        f"Expect {num_tasks}!=None when {num_classes_per_task}==None."
    )
    num_classes_per_task = num_classes // num_tasks
    return [num_classes_per_task for _ in range(num_tasks)]


def check_bijection(
    mapping: dict[int, Any] | list[Any],
    values: list[Any],
    keys: list[int] | None = None,
) -> None:
    """Warn if ``mapping`` does not form a bijection onto ``values``.

    Issues a :func:`warnings.warn` if the length of ``mapping`` differs from
    the length of ``values``. No exception is raised.

    Args:
        mapping (dict[int, Any] | list[Any]): Mapping to validate.
        values (list[Any]): Expected target values.
        keys (list[int] | None, optional): Source keys. If ``None``, integer
            indices ``0 .. len(values) - 1`` are used. (default: ``None``)
    """
    import warnings

    if keys is None:
        keys = list(range(len(values)))

    if len(mapping) != len(values):
        warnings.warn(
            f"Mapping length {len(mapping)} != values length {len(values)}. "
            + f"Mapping provides {mapping} against values {values}."
        )

    return


def reverse_mapping(mapping: dict[int, Any] | list[Any]) -> dict[Any, int]:
    """Reverse a bijection mapping from keys->values to values->keys.

    Args:
        mapping (dict[int, Any] | list[Any]): mapping to reverse

    Returns:
        dict[Any, int]: reversed mapping

    Raises:
        TypeError: if mapping is not a dict or list

    """
    reverse_lookup: dict[Any, int]
    if isinstance(mapping, (list, tuple)):
        reverse_lookup = {v: idx for idx, v in enumerate(mapping)}
    elif isinstance(mapping, dict):
        reverse_lookup = {v: idx for idx, v in mapping.items()}
    else:
        raise TypeError("mapping must be a tuple, list or dict.")

    return reverse_lookup
