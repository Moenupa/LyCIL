from typing import TypedDict


def get_or_identity(mapping: dict[str, str], key: str):
    # mapping = mapping or IDENTITY_MAPPING

    # custom key_name support with fallback `key_name`
    # 1. split names: train/val/test
    # 2. column names: input/output/image/label
    # for mapping `self.splitmapping`, key is standardized, value is customized
    return mapping.get(key, key)


class SplitMapping(TypedDict, total=False):
    train: str
    val: str
    test: str


def deterministic_shuffle(lst: list, seed: int | None) -> list:
    from numpy.random import default_rng

    default_rng(seed).shuffle(lst)
    return lst


def get_num_classes_per_task(
    num_classes_per_task: int | list[int] | None,
    num_classes: int,
    num_tasks: int | None,
) -> list[int]:
    # we want to find `N=n*tasks`, so at least one of them must be given
    if num_classes_per_task is None and num_tasks is None:
        raise ValueError("`num_classes_per_task` and `num_tasks` cannot be both None.")

    if isinstance(num_classes_per_task, list):
        return num_classes_per_task

    # if num_classes_per_task is given as int,
    # return [num_classes_per_task, ...], if length not given, try max it
    if isinstance(num_classes_per_task, int):
        num_tasks = num_tasks or (num_classes // num_classes_per_task)
        return [num_classes_per_task for _ in range(num_tasks)]

    # now num_classes_per_task is None, num_tasks must be given
    assert num_tasks is not None, (
        f"Expect {num_tasks}!=None when {num_classes_per_task}==None."
    )
    num_classes_per_task = num_classes // num_tasks
    return [num_classes_per_task for _ in range(num_tasks)]


def chunk(lst: list, n_chunks: int) -> list[list]:
    n_per_chunk = len(lst) // n_chunks

    if n_per_chunk == 0:
        raise ValueError("Number of chunks is greater than the list length.")
    if n_chunks * n_per_chunk < len(lst):
        print("WARN: some items will be dropped due to uneven chunking.")

    chunks = [lst[i * n_per_chunk : (i + 1) * n_per_chunk] for i in range(n_chunks)]
    return chunks
