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


def chunk(lst: list, n_chunks: int) -> list[list]:
    n_per_chunk = len(lst) // n_chunks

    if n_per_chunk == 0:
        raise ValueError("Number of chunks is greater than the list length.")
    if n_chunks * n_per_chunk < len(lst):
        print("WARN: some items will be dropped due to uneven chunking.")

    chunks = [lst[i * n_per_chunk : (i + 1) * n_per_chunk] for i in range(n_chunks)]
    return chunks
