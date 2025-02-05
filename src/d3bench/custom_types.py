"""Module for defining custom types and enums used in the package."""

from enum import Enum
from typing import Literal

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Criteria(Enum):
    """Enum class for benchmark criteria"""

    FUNCTIONAL = 0
    RUNTIME = 1
    CPU_RUNTIME = 2
    STORAGE = 3


class Methods(Enum):
    """Available methods for drift detection."""

    KOLMOGOROV_SMIRNOV = 0  # K-S Test
    WASSERSTEIN = 1  # Wasserstein Distance Normed
    KLD = 2  # Kullback-Leibler divergence
    PSI = 3  # Population Stability Index
    JSD = 4  # Jenson-Shannon Distance
    AD = 5  # Anderson-Darling
    CVM = 6  # Cramer-von-Mises
    HD = 7  # Hellinger distance
    MWURT = 8  # Mann-Whitney U-Rank Test
    ED = 9  # Energy-distance
    ES = 10  # Epps-Singleton
    TT = 11  # T-Test
    SPOTDIFF = 12  # Spot-The-Difference Test


class Tools(Enum):
    """Enum class for benchmark tools"""

    EVIDENTLY = 0
    NANNYML = 1
    ALIBIDETECT = 2


class Datasets(Enum):
    """Enum class for benchmark datasets"""

    ENERGY = "energy"
    OCCUPANCY = "occupancy"
