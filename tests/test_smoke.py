import importlib
import importlib.util

try:
    import pytest
except ImportError:
    pytest: None = None

"""
src
`-- lycil
    |-- __init__.py
    |-- backbone
    |-- classifier
    |-- constants.py
    |-- data
    |-- learner
    |-- metrics
    `-- scheduler
"""


def parametrize(test_func, params):
    """Custom parametrize implementation."""
    for param in params:
        if isinstance(param, tuple):
            test_func(*param)
        else:
            test_func(param)


MODULES = [
    "lycil",
    "lycil.constants",
    "lycil.data",
    "lycil.learner",
]


def import_all_modules(module_name: str):
    """Hard-coded import check for every module/submodule."""
    spec = importlib.util.find_spec(module_name)
    assert spec is not None, f"Module spec not found: {module_name}"
    module = importlib.import_module(module_name)
    assert module is not None
    assert getattr(module, "__name__", None) == module_name


if pytest is not None:

    @pytest.mark.parametrize("module_name", MODULES)
    def test_import_all_modules(module_name: str):
        import_all_modules(module_name)


if __name__ == "__main__":
    if pytest is not None:
        pytest.main()
    else:
        parametrize(import_all_modules, MODULES)
