"""
This module provides a command-line interface to run D3Bench benchmarks.

The script allows users to specify various parameters for the benchmark,
including the buildings to benchmark, the criteria to test, the tools to use,
whether to run on a VM, the dataset to use, and the logging level.
"""

import logging
from pathlib import Path

import pandas as pd
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
    output = Path("results") / options.results_file
    pd.DataFrame(results).to_csv(output, index=False)

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


# Run main function if the script is executed
if __name__ == "__main__":
    main(options=Settings())
