"""
This module contains the configuration of the datasets and tools used in the 
benchmarking process.
"""

import dataclasses as dc
import json
from pathlib import Path
from typing import Sequence, Type

from pydantic import BaseModel, Field
from pydantic.json import pydantic_encoder

from d3bench import benchmarks, config, datasets, reports, tools
from d3bench.benchmarks import Benchmark
from d3bench.config import Criteria, Datafile, Framework
from d3bench.datasets import Dataset
from d3bench.methods import BatchCD, BatchDD, OnlineCD, OnlineDD
from d3bench.tools import Tool

# pylint: disable=too-few-public-methods


# Initialize the datasets constant
DATASETS: dict[Datafile, Dataset] = {
    "energy": datasets.DataEnergy(building_id=1),
    # "occupancy": datasets.DataOccupancy(),
}

# Initialize the tools constant
TOOLS: dict[Framework, Type[Tool]] = {
    "Frouros": tools.Frouros,
    "Evidently": tools.Evidently,
    "NannyML": tools.NannyML,
    "Alibi-Detect": tools.AlibiDetect,
}


class ResultsOptions(benchmarks.Options):
    """Settings to run a benchmark and save the results."""

    criteria: set[Criteria] = Field(
        default=set(["runtime", "cputime", "memory"]),
        description="List of criteria to test.",
    )


@dc.dataclass(init=False)
class RunArgs:
    """Data class to store the data used in the benchmark."""

    criteria: set[Criteria]
    tool: Tool
    benchmark_options: benchmarks.Options

    def __init__(self, tool: Tool, options: ResultsOptions) -> None:
        self.tool = tool
        self.benchmark_options = options
        self.criteria = options.criteria


class Results(BaseModel):
    """Data class to store the results of the benchmark."""

    online_cd: list[reports.OnlineCDReport]
    online_dd: list[reports.OnlineDDReport]
    batch_cd: list[reports.BatchCDReport]
    batch_dd: list[reports.BatchDDReport]

    def __init__(self, tool: Tool, options: ResultsOptions) -> None:
        args = RunArgs(tool, options)  # Parse the arguments
        super().__init__(
            online_cd=[online_cd(m, args) for m in tool.online_cd_methods],
            online_dd=[online_dd(m, args) for m in tool.online_dd_methods],
            batch_cd=[batch_cd(m, args) for m in tool.batch_cd_methods],
            batch_dd=[batch_dd(m, args) for m in tool.batch_dd_methods],
        )

    def to_json(self) -> str:
        """Serialize results to JSON."""
        options = {"indent": 4, "default": pydantic_encoder}
        return json.dumps(self, **options)


def online_cd(method: OnlineCD, args: RunArgs) -> reports.OnlineCDReport:
    """Run an online supervised concept drift detection benchmark."""
    tool_test = args.tool.online_cd_methods[method]
    options = args.benchmark_options
    _benchmark = Benchmark(method, args.tool, tool_test, options)
    return reports.OnlineCDReport(args.criteria, _benchmark)


def online_dd(method: OnlineDD, args: RunArgs) -> reports.OnlineDDReport:
    """Run an online unsupervised concept drift detection benchmark."""
    tool_test = args.tool.online_dd_methods[method]
    options = args.benchmark_options
    _benchmark = Benchmark(method, args.tool, tool_test, options)
    return reports.OnlineDDReport(args.criteria, _benchmark)


def batch_cd(method: BatchCD, args: RunArgs) -> reports.BatchCDReport:
    """Run a batch supervised concept drift detection benchmark."""
    tool_test = args.tool.batch_cd_methods[method]
    options = args.benchmark_options
    _benchmark = Benchmark(method, args.tool, tool_test, options)
    return reports.BatchCDReport(args.criteria, _benchmark)


def batch_dd(method: BatchDD, args: RunArgs) -> reports.BatchDDReport:
    """Run a batch unsupervised concept drift detection benchmark."""
    tool_test = args.tool.batch_dd_methods[method]
    options = args.benchmark_options
    _benchmark = Benchmark(method, args.tool, tool_test, options)
    return reports.BatchDDReport(args.criteria, _benchmark)


def save_results(results: list[Results], output: str) -> None:
    """Save the results to a JSON parsed file."""
    # Generate the folder in the path if it does not exist
    output_path = config.results_path / output
    output_path.mkdir(parents=True, exist_ok=True)
    # Save the results to a file in JSON format
    for result in results:
        _save(result.online_cd, output_path / "online_cd.json")
        _save(result.online_dd, output_path / "online_dd.json")
        _save(result.batch_cd, output_path / "batch_cd.json")
        _save(result.batch_dd, output_path / "batch_dd.json")


def _save(results: Sequence[reports.Report], file_name: Path) -> None:
    options = {"indent": 4, "default": pydantic_encoder}

    # Read existing data if the file exists
    if file_name.exists():
        with open(file_name, "r", encoding="utf-8") as file:
            existing_data = json.load(file)
    else:
        existing_data = []

    # Append new results to existing data
    existing_data.extend(results)

    # Write updated data back to the file
    results_json = json.dumps(existing_data, **options)
    with open(file_name, "w", encoding="utf-8") as file:
        file.write(results_json)
        file.write("\n")
