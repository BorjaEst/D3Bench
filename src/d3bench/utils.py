"""Utility functions and classes for the drift detection methods."""

import dataclasses as dc
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any, Literal, Optional

# Define the available frameworks
Framework = Literal[
    "Frouros",
    "Evidently",
    "NannyML",
    "Alibi-Detect",
]


# Define the available methods for drift detection
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


@dc.dataclass
class Result:  # pylint: disable=too-few-public-methods
    """Result of the test method."""

    p_values: float
    drift_detected: bool
    statistic: Optional[float] = None


class BaseTestMethod(ABC):
    """Base class for the test methods."""

    @abstractmethod
    def fit(self, x_reference: Any) -> None:
        """Run the test on the reference data."""

    @abstractmethod
    def test(self, x_test: Any) -> None:
        """Run the test on the test data."""

    @abstractmethod
    def result(self) -> Result:
        """Return the result of the test."""
