import pytest

from lycil.data.util import chunk


@pytest.mark.parametrize(
    "lst, n_chunks, expected",
    [
        ([1, 2, 3, 4, 5, 6], 3, [[1, 2], [3, 4], [5, 6]]),
        ([1, 2, 3, 4, 5], 2, [[1, 2], [3, 4]]),  # last item dropped
        ([1, 2, 3], 5, None),  # error case
    ],
)
def test_chunk(lst, n_chunks, expected):
    if expected is None:
        with pytest.raises(ValueError):
            chunk(lst, n_chunks)
    else:
        result = chunk(lst, n_chunks)
        assert result == expected
