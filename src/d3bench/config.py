import os
import pathlib


# Default data path
_data_path = os.getenv("DATA_PATH", "data")
DATA_PATH = pathlib.Path(_data_path)

# Default results path
_results_path = os.getenv("RESULTS_PATH", "results")
RESULTS_PATH = pathlib.Path(_results_path)

# Default show report flag
_show_report = os.getenv("SHOW_REPORT", "true")
SHOW_REPORT = bool(_show_report)

# Data datetime filter boundaries
DATA_START_DATE = os.getenv("DATA_START_DATE", "2019-04-01")
DATA_END_DATE = os.getenv("DATA_END_DATE", "2022-04-01")
