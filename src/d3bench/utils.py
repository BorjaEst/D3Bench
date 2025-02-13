"""Utility functions and classes for the drift detection methods."""

import dataclasses as dc
import datetime as dt
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any, Literal, Optional

import numpy as np

# pylint: disable=too-many-instance-attributes
# pylint: disable=too-few-public-methods


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
class TestInformation:
    """Information about the test method used in the benchmark."""

    framework: Framework  # tool used in the benchmark
    run_on_vm: bool  # run on a VM
    repetitions: int  # number of repetitions


@dc.dataclass
class Results:  # pylint: disable=too-few-public-methods
    """Result of the test method."""

    drift_detected: Optional[bool] = None
    p_values: Optional[list[float]] = None
    statistics: Optional[list[float]] = None


@dc.dataclass
class Stats:
    """Statistics of the benchmark.

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

    avg: float  # Average value
    max: float  # Maximum value
    min: float  # Minimum value

    def __init__(self, values: list[float | int]) -> None:
        values_array = np.array(values)
        self.avg = values_array.mean()
        self.max = values_array.max()
        self.min = values_array.min()


@dc.dataclass
class DataInformation:
    """Information about the data used to benchmark."""

    len_traindata: int
    len_testdata: int


DetectorType = Literal["Concept drift", "Data drift", "Virtual drift"]
OperationType = Literal["Streaming", "Batch"]


@dc.dataclass
class DetectorInformation:
    """Information about the detector used in the benchmark."""

    multi_features: bool  # Detector supports dim>1
    fit_method: bool  # Detector has a fit method
    detector_type: DetectorType  # Detector type
    operation_type: OperationType  # Detector operation


@dc.dataclass
class Report:
    """Class to store the results of the benchmark."""

    test_information: TestInformation  # information about the test method
    method: Method  # method used in the benchmark
    detector_info: DetectorInformation  # information about the detector
    data_info: DataInformation  # information about the data used
    time: dt.datetime = dt.datetime.now()  # time of the benchmark
    runtime: Optional[Stats] = None  # runtime statistics
    cputime: Optional[Stats] = None  # runtime statistics
    ram: Optional[Stats] = None  # runtime statistics
    results: Optional[Results] = None  # results of method


class BaseTestMethod(ABC):
    """Base class for the test methods."""

    @property
    @abstractmethod
    def info(self) -> DetectorInformation:
        """Return the information of the detector."""

    @abstractmethod
    def __init__(self, n_features: int) -> None:
        """Initialize the test method."""

    @abstractmethod
    def fit(self, x_reference: Any) -> None:
        """Run the test on the reference data."""

    @abstractmethod
    def test(self, x_test: Any) -> None:
        """Run the test on the test data."""

    @abstractmethod
    def result(self) -> dict[str, Any]:
        """Return the result of the test."""
