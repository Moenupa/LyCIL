import os
from datetime import datetime

# default column name for dataset features (e.g., 'img' column for image objects)
_X_COLUMN_NAME = os.getenv("X_COLUMN_NAME", "img")

# internal column names
# _y for lycil ordered labels (not given by dataset, remapped for randomness)
_Y_COLUMN_NAME = os.getenv("Y_COLUMN_NAME", "_y")
# _cl_task_id for lycil task id, used for filtering
_CLTASK_COLUMN_NAME = os.getenv("CLTASK_COLUMN_NAME", "_cl_task_id")

# experiment identification and grouping
EXP_NAME = datetime.now().strftime("%m%d-%H%M")

# data loading constants
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "16"))
TRAIN_LOADER_KWARGS = {"batch_size": 128, "shuffle": True, "num_workers": NUM_WORKERS}
TEST_LOADER_KWARGS = {"batch_size": 128, "shuffle": False, "num_workers": NUM_WORKERS}

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
