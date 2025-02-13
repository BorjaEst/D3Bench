"""
This module provides a command-line interface to run D3Bench benchmarks.

The script allows users to specify various parameters for the benchmark,
including the buildings to benchmark, the criteria to test, the tools to use,
whether to run on a VM, the dataset to use, and the logging level.
"""

import json
import logging
from pathlib import Path

from pydantic.json import pydantic_encoder
from rich import print  # pylint: disable=redefined-builtin
from rich.logging import RichHandler

import d3bench.dataset
import d3bench.tools
from d3bench.benchmark import Benchmark, BenchmarkOptions, Report
from d3bench.config import Settings
from d3bench.dataset import Data, Dataset
from d3bench.tools import Tool

logger = logging.getLogger(__name__)

DATASETS: dict[str, Dataset] = {
    "energy": d3bench.dataset.DataEnergy(building_id=1),
    # "occupancy": d3bench.dataset.DataOccupancy(),
}

TOOLS: dict[str, Tool] = {
    "Frouros": d3bench.tools.Frouros(),
    # "Evidently": d3bench.tools.Evidently(),
    # "NannyML": d3bench.tools.NannyML(),
    # "Alibi-Detect": d3bench.tools.AlibiDetect(),
}


def main(options: Settings):
    """Run the benchmark with the given arguments."""

    # Run the benchmark with the given parameters
    print("------ Benchmark script started -------------")

    # Set the logging level from the arguments
    logging.basicConfig(
        handlers=[RichHandler(rich_tracebacks=True)],
        level=options.log_level,
    )

    # Load dataset and tools from the arguments
    logger.debug("Call arguments: %s", options)
    data = DATASETS[options.dataset].split_data()
    tools = [TOOLS[tool] for tool in options.tools]

    # Run the benchmark with the given parameters
    print("------ Benchmark execution in progress ------")
    results = run_tools(tools, data, options)

    # Print the results to the console
    print(results)

    # Save the results to a file
    save_results(results, options)

    # Print the benchmark end message
    print("---------Benchmark execution completed-------")


def run_tools(  # fmt: skip
    tools: list[Tool], data: Data, options: BenchmarkOptions
) -> list[Report]:
    """Run the benchmarks for the given tool."""

    # Run the benchmark for each tool
    results = []
    for tool in tools:
        logger.info("Running benchmarks for tool: %s", tool)
        results += list(Benchmark(tool, data, options))

    # Return the results
    return results


def save_results(
    results: list[Report], options: Settings  # fmt: skip
) -> None:
    """Save the results to a file."""
    output = Path("results") / options.results_file
    results_json = json.dumps(
        results,
        indent=4,
        default=pydantic_encoder,
    )

    # Save the results to a file in JSON format
    with open(output, "w", encoding="utf-8") as file:
        file.write(results_json)


# Run main function if the script is executed
if __name__ == "__main__":
    main(options=Settings())
