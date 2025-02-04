import os
import pathlib


# Default data path
_data_path = os.getenv("DATA_PATH", "/data")
DATA_PATH = pathlib.Path(_data_path)

# Default results path
_results_path = os.getenv("RESULTS_PATH", "/results")
RESULTS_PATH = pathlib.Path(_results_path)
