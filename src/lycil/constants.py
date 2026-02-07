import os
from datetime import datetime

# static column names in datasets for internal use
_X_COLUMN_NAME = os.getenv("X_COLUMN_NAME", "_x")
_Y_COLUMN_NAME = os.getenv("Y_COLUMN_NAME", "_y")
_CLTASK_COLUMN_NAME = os.getenv("CLTASK_COLUMN_NAME", "_cl_task_id")

# experiment identification and grouping
_EXP_NAME = datetime.now().strftime("exp_%Y%m%d_%H%M%S")

if os.getenv("HF_DATASETS_DISABLE_PROGRESS_BARS") == "":
    # disable datasets progress bars if not explicitly set
    os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"


def get_seed() -> int | None:
    if seed := os.getenv("PL_GLOBAL_SEED"):
        return int(seed)

    return None
