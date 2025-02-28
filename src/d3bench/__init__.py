"""
This module contains the configuration of the datasets and tools used in the 
benchmarking process.
"""

import json
from pathlib import Path
from typing import Callable, Type

from pydantic import BaseModel, Field
from pydantic.json import pydantic_encoder

from d3bench import benchmarks, config, datasets, reports, tools
from d3bench.benchmarks import Benchmark
from d3bench.config import Criteria, Datafile, Framework, Method
from d3bench.datasets import Dataset
from d3bench.tools import Tool

# Initialize the datasets constant
DATASETS: dict[Datafile, Dataset] = {
    "energy": datasets.DataEnergy(building_id=1),
    # "occupancy": datasets.DataOccupancy(),
}

# Initialize the tools constant
TOOLS: dict[Framework, Type[Tool]] = {
    "Frouros": tools.Frouros,
    # "Evidently": tools.Evidently,
    # "NannyML": tools.NannyML,
    # "Alibi-Detect": tools.AlibiDetect,
}


class ResultsOptions(benchmarks.Options):
    """Settings to run a benchmark and save the results."""

    criteria: set[Criteria] = Field(
        default=set(["functional", "runtime", "cputime", "memory"]),
        description="List of criteria to test.",
    )


class Results(BaseModel):
    """Data class to store the results of the benchmark."""

    online_cd: list[reports.OnlineCDReport]
    online_dd: list[reports.OnlineDDReport]
    batch_cd: list[reports.BatchCDReport]
    batch_dd: list[reports.BatchDDReport]

    def __init__(self, tool: Tool, options: ResultsOptions) -> None:
        super().__init__(
            online_cd=[
                online_cd_report(options.criteria, tool, method, options)
                for method in tool.online_cd_methods
            ],
            online_dd=[],  # TODO: add the rest of the reports
            batch_cd=[],  # TODO: add the rest of the reports
            batch_dd=[],  # TODO: add the rest of the reports
        )

    def to_json(self) -> str:
        """Serialize results to JSON."""
        options = {"indent": 4, "default": pydantic_encoder}
        return json.dumps(self, **options)


def online_cd_report(
    criteria: set[Criteria],
    tool: Tool,
    method: Method,
    options: benchmarks.Options,
) -> reports.OnlineCDReport:
    """Run an online supervised concept drift detection benchmark."""
    _benchmark = Benchmark(tool, method, options)
    return reports.OnlineCDReport(criteria, _benchmark)


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


def _save(results: list[reports.Report], file_name: Path) -> None:
    options = {"indent": 4, "default": pydantic_encoder}
    results_json = json.dumps(results, **options)
    with open(file_name, "a", encoding="utf-8") as file:
        file.write(results_json)
        file.write("\n")
