import dataclasses as dc
import datetime as dt
import logging
import time
import timeit
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
from memory_profiler import memory_usage
from pydantic import Field

from d3bench.dataset import Data
from d3bench.tool import Framework, Method, Tool

# pylint: disable=too-few-public-methods


logger = logging.getLogger(__name__)
Criteria = Literal["FUNCTIONAL", "RUNTIME", "CPUTIME", "MEMORY"]


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
        building_id: int,
        tool: Tool,
        data: Data,
        settings: Optional[BenchmarkOptions] = None,
    ):
        settings = settings or BenchmarkOptions()
        self.building_id = building_id
        self.tool = tool
        self.data = data
        self.criteria = settings.criteria
        self.on_vm = settings.vm

    def __call__(self, method: Method):
        """Run the benchmark with the given parameters."""

        # Prepare the job for the benchmark
        logger.debug("Prep. benchmark job %s", (method, self.tool))
        job = Job(self.tool, method, self.data)

        # Train the detector before running the benchmark
        try:  # Raise NotImplementedError if tool has no training call
            job.fit()  # train the detector
        except NotImplementedError:
            logger.debug("Skipping training for %s", self.tool)

        # Prepare report for the benchmark
        logger.debug("Prep. benchmark report")
        framework = self.tool.name  # tool used in the benchmark
        report = Report(framework, method, self.building_id, self.on_vm)

        # Run the benchmark for each criterion
        logger.debug("Running benchmark for %s", self.criteria)
        for criteria in self.criteria:
            criteria_fn[criteria](job, report)

        # Return the results as report
        logger.debug("Benchmark completed")
        return report


class Job:
    """Class to run a benchmark job with the given parameters."""

    def __init__(self, tool: Tool, method: Method, data: Data):
        # Prepare the job for the benchmark, copy to avoid side effects
        self.job_store: dict[str, Any] = {}
        self.tool = tool
        self.method = method
        tool.setup(method, store=self.job_store)
        self.data = {
            "reference": tool.preprocess(data[0].copy(), self.job_store),
            "test": tool.preprocess(data[1].copy(), self.job_store),
        }

    def fit(self) -> None:
        """Run the benchmark with the given parameters."""
        self.tool.fit(self.data["reference"], self.job_store)

    def test(self) -> None:
        """Run the benchmark with the given parameters."""
        self.tool.test(self.data["test"], self.job_store)

    @property
    def report(self) -> dict[str, Any]:
        """Return the report of the benchmark."""
        return self.tool.postprocess(self.job_store)


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
    building_id: int  # building ID used in the benchmark
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
    drift_score: Optional[float] = None
    drift_detected: Optional[bool] = None
    detection_stats: dict = dc.field(default_factory=dict)

    def __repr__(self) -> str:
        # TODO: improve with rich
        return f"{self.__class__.__name__}({self.__dict__})"

    def as_dataframe(self) -> pd.DataFrame:
        """Convert the report to a DataFrame."""
        return pd.DataFrame.from_dict(self.__dict__)


def run_functional(job: Job, report: Report):
    """
    Run the benchmark for the FUNCTIONAL criterion.
    Includes the drift detection statistics in the report.
    """

    # Run the drift detection and store the results
    job.test()  # run the test
    report.detection_stats.update(job.report)

    # Save report of drift detection
    pass  # TODO: save the report


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
