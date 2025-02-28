"""Module to run a benchmark to obtain Results."""

import logging
import time
import timeit
from typing import Any, Optional

import pandas as pd
from memory_profiler import memory_usage
from pydantic import Field

from d3bench import tools, utils
from d3bench.config import Criteria, Data, Method
from d3bench.tools import Tool
from pydantic_settings import BaseSettings

# pylint: disable=too-few-public-methods


logger = logging.getLogger(__name__)


class Options(BaseSettings):
    """Settings to run a benchmark."""

    repetitions: int = Field(
        default=3,
        description="Number of repetitions for the benchmark.",
    )


class Benchmark:
    """Class to run a benchmark to obtain Results."""

    def __init__(
        self,
        tool: Tool,
        method: Method,
        options: Optional[Options] = None,
    ) -> None:
        options = options or Options()
        self.repetitions = options.repetitions
        self.tool = tool
        self.method = method
        self.x_reference = tool.data["x_reference"].copy()
        self.x_test = tool.data["x_test"].copy()
        self.job = Job(method, self.data)
        self.job.fit()  # Fit the model

    @property
    def data(self) -> Data:
        """Return the data used in the benchmark."""
        return {
            "x_reference": self.x_reference,
            "x_test": self.x_test,
        }

    def get_runtimes(self) -> list[float]:
        """
        Run the benchmark for the RUNTIME criterion.
        Measure elapsed time using wall-clock time in milliseconds.
        Includes waiting time for resources.
        """

        # Create a runtime timer
        # TODO: Future implementation for time training phase
        timer = timeit.Timer(self.job.test, timer=time.time)

        # Time runtimes measurements
        return timer.repeat(self.repetitions, number=1)

    def get_cputimes(self) -> list[float]:
        """
        Run the benchmark for the CPUTIME criterion.
        Measures CPU resources consumed by the process (user and system)
        (exclude: waiting time for resources): time in ms
        """

        # Create a runtime timer
        # TODO: Future implementation for time training phase
        timer = timeit.Timer(self.job.test, timer=time.process_time)

        # Time runtimes measurements
        return timer.repeat(self.repetitions, number=1)

    def get_memories(self) -> list[float]:
        """
        Run the benchmark for the MEMORY criterion.
        Measures RAM resources consumed by the process (user and system)
        """

        # Run the drift detection for each building
        # TODO: Future implementation for time training phase
        repeat = range(self.repetitions)
        rmem = [memory_usage(self.job.test) for _ in repeat]

        # Memory in run as the maximum memory used during the run
        return [max(mem) for mem in rmem]


class Job:
    """Class to run a benchmark job with the given parameters."""

    def __init__(self, method: Method, data: Data) -> None:
        # Prepare the job for the benchmark, copy to avoid side effects
        self.x_reference = data["x_reference"]
        self.x_test = data["x_test"]
        self.features = self.x_reference.columns
        self.detector = method(self.features)

    def fit(self) -> None:
        """Run the benchmark with the given parameters."""
        self.detector.fit(self.x_reference)

    def test(self) -> None:
        """Run the benchmark with the given parameters."""
        self.detector.test(self.x_test)

    @property
    def results(self) -> dict[str, Any]:
        """Return the results of the benchmark."""
        return self.detector.result()
