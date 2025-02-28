"""Module to run a benchmark to obtain Results."""

import logging
import time
import timeit
from typing import Any, Optional, Type, Union

from memory_profiler import memory_usage
from pydantic import Field
from pydantic_settings import BaseSettings

from d3bench.methods import BatchCD, BatchDD, OnlineCD, OnlineDD
from d3bench.tools import Tool
from d3bench.utils import BaseTestMethod, Data

# pylint: disable=too-few-public-methods
logger = logging.getLogger(__name__)


class Options(BaseSettings):
    """Settings to run a benchmark."""

    repetitions: int = Field(
        default=3,
        description="Number of repetitions for the benchmark.",
    )

    on_vm: bool = Field(
        default=False,
        description="Flag to run the benchmark on a VM.",
    )


class Benchmark:
    """Class to run a benchmark to obtain Results."""

    def __init__(
        self,
        method: Union[OnlineCD, OnlineDD, BatchCD, BatchDD],
        tool: Tool,
        test: Type[BaseTestMethod],
        options: Optional[Options] = None,
    ) -> None:
        options = options or Options()
        self.repetitions = options.repetitions
        self.run_on_vm = options.on_vm
        self.method = method
        self.tool = tool
        self.test = test
        self.job = Job(self)
        self.fit_job()

    @property
    def data(self) -> Data:
        """Return the data used in the benchmark."""
        return self.tool.data

    def fit_job(self) -> None:
        """Fit the job for the benchmark."""
        try:
            self.job.fit()
        except NotImplementedError:
            logger.debug("No train method for %s", self.method)

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

    def __init__(self, benchmark: Benchmark) -> None:
        self.benchmark = benchmark
        self.detector = benchmark.test(benchmark.tool.data.features)

    def fit(self) -> None:
        """Run the benchmark with the given parameters."""
        # Call tool.reference_data to ensure preprocessing
        self.detector.fit(self.benchmark.tool.reference_data)  # Cached

    def test(self) -> None:
        """Run the benchmark with the given parameters."""
        # Call tool.testing_data to ensure preprocessing
        self.detector.test(self.benchmark.tool.testing_data)

    @property
    def results(self) -> dict[str, Any]:
        """Return the results of the benchmark."""
        return self.detector.result()
