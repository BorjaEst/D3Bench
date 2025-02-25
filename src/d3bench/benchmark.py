"""Module to run a benchmark to obtain Results."""

import logging
import time
import timeit
from typing import Any, Optional

from memory_profiler import memory_usage
from pydantic import Field

from d3bench import utils
from d3bench.config import Criteria, Method
from d3bench.dataset import Data
from d3bench.tools import Tool
from d3bench.utils import Report

# pylint: disable=too-few-public-methods


logger = logging.getLogger(__name__)


class BenchmarkOptions:
    """Settings to instantiate a benchmark."""

    criteria: set[Criteria] = Field(
        default=set(["RUNTIME", "CPUTIME", "MEMORY"]),
        description="List of criteria to test.",
    )
    repetitions: int = Field(
        default=3,
        description="Number of repetitions to run the benchmark.",
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
        self.criteria = settings.criteria
        self.tool = tool
        self.x_reference = data[0]
        self.x_test = data[1]
        self.run_on_vm = settings.vm
        self.repetitions = settings.repetitions

    def __iter__(self):
        """Run the benchmark with the given parameters."""
        for method in self.tool.methods:
            logger.debug("Running benchmark for %s", method)
            yield self(method)

    @property
    def test_info(self) -> utils.TestInformation:
        """Return the information of the test method used in the benchmark."""
        return utils.TestInformation(
            framework=self.tool.name,
            run_on_vm=self.run_on_vm,
            repetitions=self.repetitions,
        )

    @property
    def data_info(self) -> utils.DataInformation:
        """Return the information of the data used in the benchmark."""
        return utils.DataInformation(
            len_traindata=self.x_reference.shape[0],
            len_testdata=self.x_test.shape[0],
        )

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
        report = utils.Report(
            test_information=self.test_info,
            method=method,
            detector_info=job.detector.info,
            data_info=self.data_info,
        )

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
        self.detector = tool[method](features=self.x_reference.columns)

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
    report.results = utils.Results(**job.results)


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
    n_repeat = report.test_information.repetitions
    runtimes = timer.repeat(n_repeat, number=1)

    # Compute statistics
    report.runtime = utils.Stats(runtimes)


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
    n_repeat = report.test_information.repetitions
    runtimes = timer.repeat(n_repeat, number=1)

    # Compute statistics
    report.cputime = utils.Stats(runtimes)


def run_memory(job: Job, report: Report):
    """
    Run the benchmark for the MEMORY criterion.
    Measures RAM resources consumed by the process (user and system)
    """

    # Run the drift detection for each building
    # TODO: Future implementation for time training phase
    n_repeat = report.test_information.repetitions
    rmem = [memory_usage(job.test) for _ in range(n_repeat)]

    # Memory in run as the maximum memory used during the run
    run_memories = [max(mem) for mem in rmem]

    # Compute statistics
    report.ram = utils.Stats(run_memories)


criteria_fn = {
    "FUNCTIONAL": run_functional,
    "RUNTIME": run_runtime,
    "CPUTIME": run_cputime,
    "MEMORY": run_memory,
}


def run_tools(
    tools: list[Tool],
    data: Data,
    args: BenchmarkOptions,
) -> list[Report]:
    """Run the benchmarks for the given tool."""

    # Run the benchmark for each tool
    results = []
    for tool in tools:
        logger.info("Running benchmarks for tool: %s", tool)
        results += list(Benchmark(tool, data, args))

    # Return the results
    return results
