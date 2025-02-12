"""Utility functions and classes for the drift detection methods."""

import dataclasses as dc
import datetime as dt
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any, Literal, Optional

import pandas as pd

# Define the available frameworks
Framework = Literal[
    "Frouros",
    "Evidently",
    "NannyML",
    "Alibi-Detect",
]

# Define the available datasets
Dataset = Literal[
    "energy",
    "occupancy",
]

# Define the available criteria to test
Criteria = Literal[
    "FUNCTIONAL",
    "RUNTIME",
    "CPUTIME",
    "MEMORY",
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
class Report:
    """Class to store the results of the benchmark.

    Note It’s tempting to calculate mean and standard deviation from the
    result vector and report these. However, this is not very useful. In
    a typical case, the lowest value gives a lower bound for how fast your
    machine can run the given code snippet; higher values in the result
    vector are typically not caused by variability in Python’s speed, but
    by other processes interfering with your timing accuracy. So the min()
    of the result is probably the only number you should be interested in.

    After that, you should look at the entire vector and apply common sense
    rather than statistics.
    """

    framework: Framework  # tool used in the benchmark
    test_method: Method  # method used in the benchmark
    len_testdata: int  # number of points in the test dataset
    run_on_vm: bool = False  # run on a VM
    time: dt.datetime = dt.datetime.now()  # time of the benchmark
    repetitions: int = 10  # number of repetitions
    runtime_avg: Optional[float] = None
    runtime_max: Optional[float] = None
    runtime_min: Optional[float] = None
    cputime_avg: Optional[float] = None
    cputime_max: Optional[float] = None
    cputime_min: Optional[float] = None
    ram_avg: Optional[float] = None
    ram_max: Optional[float] = None
    ram_min: Optional[float] = None
    drift_detected: Optional[bool] = None
    p_value: Optional[float] = None
    statistic: Optional[float] = None

    def __repr__(self) -> str:
        # TODO: improve with rich
        return f"{self.__class__.__name__}({self.__dict__})"

    def as_dataframe(self) -> pd.DataFrame:
        """Convert the report to a DataFrame."""
        return pd.DataFrame.from_dict(self.__dict__)


@dc.dataclass
class Result:  # pylint: disable=too-few-public-methods
    """Result of the test method."""

    drift_detected: Optional[bool] = None
    p_value: Optional[float] = None
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
