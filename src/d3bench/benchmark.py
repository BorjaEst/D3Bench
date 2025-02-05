import dataclasses as dc
import datetime as dt
import os.path
import time
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from memory_profiler import memory_usage
from pydantic import Field

from d3bench.dataset import Dataset
from d3bench.tool import Methods, Tool

# pylint: disable=too-few-public-methods


Criteria = Literal["FUNCTIONAL", "RUNTIME", "CPUTIME", "MEMORY"]


RESULTS_PATH = Path("results")


class BenchmarkOptions:
    """Settings to instantiate a benchmark."""

    report_id: str = Field(
        default=f"report_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        description="Report ID.",
    )
    buildings: set[int] = Field(
        default={1},
        description="List of building IDs to benchmark.",
    )
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
        method: Methods,
        ds: Dataset,
        settings: Optional[BenchmarkOptions] = None,
    ):
        settings = settings or BenchmarkOptions()
        self.tool = tool  # tool to use in the benchmark
        self.method = method  # method to use in the benchmark
        self.dataset = ds  # dataset to use in the benchmark
        self.report_dir = RESULTS_PATH / settings.report_id
        self.criteria = settings.criteria
        self.buildings = settings.buildings
        self.on_vm = settings.vm

    def run(self):
        """Run the benchmark with the given parameters."""

        # Run the benchmark for each criterion
        report = Report()
        for criteria in self.criteria:
            criteria_fn[criteria](self, report)

        # Return the results as report
        return report


@dc.dataclass
class Report:
    """Class to store the results of the benchmark."""

    detection_stats: dict = dc.field(default_factory=dict)
    runtime_avg: Optional[float] = None
    runtime_max: Optional[float] = None
    cputime_avg: Optional[float] = None
    cputime_max: Optional[float] = None
    ram_avg: Optional[float] = None
    ram_max: Optional[float] = None

    def __repr__(self) -> str:
        return (
            "Report(\n"
            f"\t Detection Statistics: {self.detection_stats}, \n"
            f"\t Runtime AVG: {self.runtime_avg:.8f}, \n"
            f"\t Runtime MAX: {self.runtime_max:.8f}, \n"
            f"\t CPU Time AVG:{self.cputime_avg:.8f}, \n"
            f"\t CPU Time MAX: {self.cputime_max:.8f}, \n"
            f"\t RAM Usage AVG: {self.ram_avg:.8f}, \n"
            f"\t RAM Usage MAX: {self.ram_max:.8f}, \n"
            ")"
        )

    def as_dataframe(self) -> pd.DataFrame:
        """Convert the report to a DataFrame."""
        return pd.DataFrame.from_dict(self.__dict__)


def run_functional(benchmark: Benchmark, report: Report):
    """
    Run the benchmark for the FUNCTIONAL criterion.
    Includes the drift detection statistics in the report.
    """

    # Run the drift detection for each building
    for building_id in benchmark.buildings:

        # Split into reference and current dataset
        df_train, df_test = benchmark.dataset.splitTrainTest(building_id)
        report.detection_stats[building_id] = {}

        # runDriftDetection without report generation
        result = benchmark.tool(df_train, df_test, benchmark.method)
        report.detection_stats[building_id].update(result)


def run_runtime(benchmark: Benchmark, report: Report):
    """
    Run the benchmark for the RUNTIME criterion.
    Measure elapsed time using wall-clock time in milliseconds.
    Includes waiting time for resources.
    """

    # Run the drift detection for each building
    runtimes = []
    for building_id in benchmark.buildings:

        # Split into training and test
        df_train, df_test = benchmark.dataset.splitTrainTest(building_id)
        st = time.time()  # TODO: timeit might be better

        # Timestamp before executing drift detection
        _ = benchmark.tool(df_train, df_test, building_id)

        # Timestamp after executing, compute runtime in ms
        runtimes.append((time.time() - st) * 1000)

    # compute average and max runtimes
    report.runtime_avg = sum(runtimes) / len(runtimes)
    report.runtime_max = max(runtimes)


def run_cputime(benchmark: Benchmark, report: Report):
    """
    Run the benchmark for the CPUTIME criterion.
    Measures CPU resources consumed by the process (user and system)
    (exclude: waiting time for resources): time in ms
    """

    # Run the drift detection for each building
    runtimes = []
    for building_id in benchmark.buildings:

        # Split into training and test
        df_train, df_test = benchmark.dataset.splitTrainTest(building_id)
        st = time.process_time()

        # Start measuring CPU Usage (include user and system cpu time)
        _ = benchmark.tool(df_train, df_test, benchmark.method)

        # End measuring CPU Usage, compute cpu in GB
        runtimes.append((time.process_time() - st) * 1000)

    # compute average and max runtimes
    report.cputime_avg = sum(runtimes) / len(runtimes)
    report.cputime_max = max(runtimes)


def run_memory(benchmark: Benchmark, report: Report):
    """
    Run the benchmark for the MEMORY criterion.
    Measures RAM resources consumed by the process (user and system)
    """

    # Run the drift detection for each building
    run_memories = []
    for building_id in benchmark.buildings:

        # Split into training and test
        df_train, df_test = benchmark.dataset.splitTrainTest(building_id)

        # Start measuring CPU Usage (include user and system cpu time)
        args = (df_train, df_test, benchmark.method)
        memory = memory_usage((benchmark.tool, args))

        # End measuring CPU Usage, compute cpu in GB
        run_memories.append(memory)

    # compute average and max run_memories
    report.ram_avg = sum(run_memories) / len(run_memories)
    report.ram_max = max(run_memories)


criteria_fn = {
    "FUNCTIONAL": run_functional,
    "RUNTIME": run_runtime,
    "CPUTIME": run_cputime,
    "MEMORY": run_memory,
}


def print_report(benchmark: Benchmark, report: Report):
    """Print the benchmark report to the console."""

    # Print the benchmark data
    print("==============================")
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    name = benchmark.tool.name
    if benchmark.tool.show_report:
        name = name + " with report"
    print(f"Benchmark Report: {name}")
    print(f"Report generated at: {current_time}")
    print("==============================")

    # Print the drift detection statistics
    df_stats = pd.DataFrame.from_dict(report.detection_stats)
    # TODO: print the drift detection statistics

    # Print the runtime statistics
    print("==============================")
    print(report)
    print("==============================")


def save_report(benchmark: Benchmark, report: Report):
    """Save the benchmark report to a CSV file."""

    # Calculate the path from the benchmark settings
    os.makedirs(benchmark.report_dir, exist_ok=True)
    report_path = benchmark.report_dir / "benchmark_report.csv"

    # Create DataFrames from the drift detection statistics
    df_report = report.as_dataframe()

    # Save the report to a CSV file
    if os.path.exists(report_path):
        df_report.to_csv(report_path, mode="a", index=False, header=False)
    else:
        df_report.to_csv(report_path, index=False)
