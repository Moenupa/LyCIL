import os

# static column names in datasets for internal use
_X_COLUMN_NAME = os.getenv("X_COLUMN_NAME", "_x")
_Y_COLUMN_NAME = os.getenv("Y_COLUMN_NAME", "_y")
_CLTASK_COLUMN_NAME = os.getenv("CLTASK_COLUMN_NAME", "_cl_task_id")
