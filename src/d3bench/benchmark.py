"""Module to run a benchmark to obtain Results."""

import logging
import time
import timeit
from pathlib import Path
from typing import Optional

import numpy as np
from memory_profiler import memory_usage
from pydantic import Field

from d3bench.dataset import Data
from d3bench.tools import Tool
from d3bench.utils import Criteria, Method, Report, Result

# pylint: disable=too-few-public-methods


logger = logging.getLogger(__name__)
RESULTS_PATH = Path("results")


class BenchmarkOptions:
    """Settings to instantiate a benchmark."""

    criteria: set[Criteria] = Field(
        default=set(["RUNTIME", "CPUTIME", "MEMORY"]),
        description="List of criteria to test.",
    )
    vm: bool = Field(
        default=False,
        description="Run on a VM if set.",
    )


class Benchmark:
    """Class to run a benchmark to obtain Results."""

    def __init__(
        self,
        tool: Tool,
        data: Data,
        settings: Optional[BenchmarkOptions] = None,
    ):
        settings = settings or BenchmarkOptions()
        self.tool = tool
        self.x_reference = data[0]
        self.x_test = data[1]
        self.criteria = settings.criteria
        self.on_vm = settings.vm

    def __iter__(self):
        """Run the benchmark with the given parameters."""
        for method in self.tool.methods:
            logger.debug("Running benchmark for %s", method)
            yield self(method)

    def __call__(self, method: Method):
        """Run the benchmark with the given parameters."""

        # Prepare the job for the benchmark
        logger.debug("Prep. benchmark job %s", (method, self.tool))
        job = Job(self.tool, method, benchmark=self)

        # Train the detector before running the benchmark
        try:  # Raise NotImplementedError if tool has no training call
            job.fit()  # train the detector
        except NotImplementedError:
            logger.debug("Skipping training for %s", self.tool)

        # Prepare report for the benchmark
        logger.debug("Prep. benchmark report")
        framework = self.tool.name  # tool used in the benchmark
        n_points = self.x_reference.shape[0]  # number of points in the dataset
        report = Report(framework, method, n_points, self.on_vm)

        # Run the benchmark for each criterion
        logger.debug("Running benchmark for %s", self.criteria)
        for criteria in self.criteria:
            criteria_fn[criteria](job, report)

        # Return the results as report
        logger.debug("Benchmark completed")
        return report


class Job:
    """Class to run a benchmark job with the given parameters."""

    def __init__(  # fmt: skip
        self, tool: Tool, method: Method, benchmark: Benchmark
    ) -> None:
        # Prepare the job for the benchmark, copy to avoid side effects
        self.x_reference = tool.preprocess(benchmark.x_reference.copy())
        self.x_test = tool.preprocess(benchmark.x_test.copy())
        self.benchmark = benchmark
        self.detector = tool[method]()

    def fit(self) -> None:
        """Run the benchmark with the given parameters."""
        self.detector.fit(self.x_reference)

    def test(self) -> None:
        """Run the benchmark with the given parameters."""
        self.detector.test(self.x_test)

    @property
    def result(self) -> Result:
        """Return the result of the test."""
        return self.detector.result()


def run_functional(job: Job, report: Report):
    """
    Run the benchmark for the FUNCTIONAL criterion.
    Includes the drift detection statistics in the report.
    """

    # Run the drift detection and store the results
    job.test()  # run the test

    # Save report of drift detection
    pass  # TODO: save the report

    # Collect the results of the drift detection
    result = job.result
    report.drift_detected = result.drift_detected
    report.p_value = result.p_value
    report.statistic = result.statistic


def run_runtime(job: Job, report: Report):
    """
    Run the benchmark for the RUNTIME criterion.
    Measure elapsed time using wall-clock time in milliseconds.
    Includes waiting time for resources.
    """

    # Create a runtime timer
    # TODO: Future implementation for time training phase
    timer = timeit.Timer(job.test, timer=time.time)

    # Time runtimes measurements
    _runtimes = timer.repeat(repeat=report.repetitions, number=1)
    runtimes = np.array(_runtimes, dtype=float)

    # Compute statistics
    report.runtime_avg = runtimes.mean()
    report.runtime_max = runtimes.max()
    report.runtime_min = runtimes.min()


def run_cputime(job: Job, report: Report):
    """
    Run the benchmark for the CPUTIME criterion.
    Measures CPU resources consumed by the process (user and system)
    (exclude: waiting time for resources): time in ms
    """

    # Create a runtime timer
    # TODO: Future implementation for time training phase
    timer = timeit.Timer(job.test, timer=time.process_time)

    # Time runtimes measurements
    _runtimes = timer.repeat(repeat=report.repetitions, number=1)
    runtimes = np.array(_runtimes, dtype=float)

    # Compute statistics
    report.cputime_avg = runtimes.mean()
    report.cputime_max = runtimes.max()
    report.cputime_min = runtimes.min()


def run_memory(job: Job, report: Report):
    """
    Run the benchmark for the MEMORY criterion.
    Measures RAM resources consumed by the process (user and system)
    """

    # Run the drift detection for each building
    # TODO: Future implementation for time training phase
    rmem = [memory_usage(job.test) for _ in range(report.repetitions)]

    # Memory in run as the maximum memory used during the run
    run_memories = np.array([max(mem) for mem in rmem], dtype=float)

    # Compute statistics
    report.ram_avg = run_memories.mean()
    report.ram_max = run_memories.max()
    report.ram_min = run_memories.min()


criteria_fn = {
    "FUNCTIONAL": run_functional,
    "RUNTIME": run_runtime,
    "CPUTIME": run_cputime,
    "MEMORY": run_memory,
}
