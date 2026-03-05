def test_constants_import():
    from .constants import TEST_LOADER_KWARGS, VAL_LOADER_KWARGS

    assert VAL_LOADER_KWARGS != {}
    assert TEST_LOADER_KWARGS != {}
