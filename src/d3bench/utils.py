"""Utility functions and classes for the drift detection methods."""

import datetime as dt
import json
from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel
from pydantic.json import pydantic_encoder

from d3bench import config

# pylint: disable=too-many-instance-attributes
# pylint: disable=too-few-public-methods


class TestInformation(BaseModel):
    """Information about the test method used in the benchmark."""

    framework: config.Framework  # tool used in the benchmark
    run_on_vm: bool  # run on a VM
    repetitions: int  # number of repetitions


class Results(BaseModel):
    """Result of the test method."""

    dataset_drift: Optional[bool] = None
    p_values: Optional[dict[str, float]] = None
    statistics: Optional[dict[str, float]] = None


class Stats(BaseModel):
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
        super().__init__()  # Call pydantic initializer
        values_array = np.array(values)
        self.avg = float(values_array.mean())
        self.max = float(values_array.max())
        self.min = float(values_array.min())


class DetectorInformation(BaseModel):
    """Information about the detector used in the benchmark."""

    multivariate_detector: bool  # Detector supports dim>1
    fit_method: bool  # Detector has a fit method
    detector_type: config.DetectorType  # Detector type
    operation_type: config.OperationType  # Detector operation


class DataInformation(BaseModel):
    """Information about the data used to benchmark."""

    len_traindata: int
    len_testdata: int


class Report(BaseModel):
    """Class to store the results of the benchmark."""

    test_information: TestInformation  # information about the test method
    method: config.Method  # method used in the benchmark
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


def save_results(
    results: list[Report],
    file_name: str,
) -> None:
    """Save the results to a JSON parsed file."""
    output = config.results_path / file_name
    results_json = json.dumps(
        results,
        indent=4,
        default=pydantic_encoder,
    )

    # Save the results to a file in JSON format
    with open(output, "w", encoding="utf-8") as file:
        file.write(results_json)
