"""Configuration settings for the d3bench package."""

import os
from enum import Enum, StrEnum
from pathlib import Path
from typing import Literal, Tuple, TypeAlias

import pandas as pd

DATA_PATH = os.getenv("DATA_PATH", "data")
data_path = Path(DATA_PATH)


RESULTS_PATH = os.getenv("RESULTS_PATH", "results")
results_path = Path(RESULTS_PATH)


class Method(StrEnum):
    """Available methods for drift detection."""

    KOLMOGOROV_SMIRNOV = "K-S Test"
    WASSERSTEIN = "Wasserstein Distanz"
    KLD = "K-L Divergence"
    PSI = "PSI"
    JSD = "J-S Distance"
    AD = "Anderson-Darling"
    CVM = "Cramer-von-Mises"
    HD = "Hellinger-Distance"
    MWURT = "Mann-Whitney U-Rank Test"
    ED = "Energy-Distance"
    ES = "Epps-Singleton"
    TT = "T-Test"
    SPOTDIFF = "Spot-The-Difference Test"


class Datasets(Enum):
    """Enum class for benchmark datasets"""

    ENERGY = "energy"
    OCCUPANCY = "occupancy"


Framework: TypeAlias = Literal[
    "Frouros",
    "Evidently",
    "NannyML",
    "Alibi-Detect",
]

Data = Tuple[pd.DataFrame, pd.DataFrame]
Dataset: TypeAlias = Literal[
    "energy",
    "occupancy",
]


Criteria: TypeAlias = Literal[
    "FUNCTIONAL",
    "RUNTIME",
    "CPUTIME",
    "MEMORY",
]


DetectorType: TypeAlias = Literal[
    "Concept drift",
    "Data drift",
    "Virtual drift",
]
OperationType: TypeAlias = Literal[
    "Streaming",
    "Batch",
]
