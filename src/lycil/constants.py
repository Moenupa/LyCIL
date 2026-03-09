import os
from datetime import datetime

# static column names in datasets for internal use
_X_COLUMN_NAME = os.getenv("X_COLUMN_NAME", "img")
_Y_COLUMN_NAME = os.getenv("Y_COLUMN_NAME", "_y")
_CLTASK_COLUMN_NAME = os.getenv("CLTASK_COLUMN_NAME", "_cl_task_id")

# experiment identification and grouping
_EXP_NAME = datetime.now().strftime("exp_%Y%m%d_%H%M%S")

if os.getenv("HF_DATASETS_DISABLE_PROGRESS_BARS") is None:
    # disable datasets progress bars if not explicitly set
    os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"


def get_seed() -> int | None:
    """Return the Lightning seed from the environment, if available.

    Returns:
        int | None: Integer seed parsed from the ``PL_GLOBAL_SEED`` environment
            variable, or ``None`` if the variable is not set.
    """
    if seed := os.getenv("PL_GLOBAL_SEED"):
        return int(seed)

    return None


def is_env_enabled(env_var: str, default: str = "0") -> bool:
    """Check whether an environment variable is set to a truthy value.

    Truthy values are ``"true"``, ``"yes"``, ``"on"``, ``"t"``, ``"y"``, ``"1"``
    (case-insensitive).

    Args:
        env_var (str): Name of the environment variable to inspect.
        default (str, optional): Value to use when the variable is unset.
            (default: ``"0"``)

    Returns:
        bool: ``True`` if the resolved value is truthy, ``False`` otherwise.
    """
    return os.getenv(env_var, default).lower() in ["true", "yes", "on", "t", "y", "1"]
