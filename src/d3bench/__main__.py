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

from d3bench.benchmark import Benchmark, Report
from d3bench.config import RunSettings, Settings
from d3bench.dataset import Data, DataEnergy, DataOccupancy, Dataset
from d3bench.tool import AlibiDetect, Evidently, NannyML, Tool

logger = logging.getLogger(__name__)

DATASETS = {
    "energy": DataEnergy(),
    "occupancy": DataOccupancy(),
}

TOOLS = {
    "Evidently": Evidently(),
    "NannyML": NannyML(),
    # "Alibi-Detect": AlibiDetect(),
}


def main(options: Settings):
    """Run the benchmark with the given arguments."""

    # Run the benchmark with the given parameters
    print("------ Benchmark script started -------------")

    # Set the logging level from the arguments
    logger.setLevel(options.log_level)
    logger.debug("args: %s", options)

    # Load dataset and tools from the arguments
    buildings = options.buildings
    dataset = DATASETS[options.dataset]
    tools = [TOOLS[tool] for tool in options.tools]

    # Run the benchmark with the given parameters
    print("------ Benchmark execution in progress ------")
    results = run_buildings(buildings, tools, dataset, options)

    # Print the results to the console
    print(results)

    # Save the results to a file
    output = Path("results") / options.results_file
    pd.DataFrame(results).to_csv(output, index=False)

    # Print the benchmark end message
    print("---------Benchmark execution completed-------")


def run_buildings(
    buildings: set[int], tools: list[Tool], ds: Dataset, options: RunSettings
) -> list[Report]:
    """Run the benchmarks for the given buildings."""

    # Run the benchmark for each building
    results = []
    for building_id in buildings:
        logger.info("Running benchmarks for building: %s", building_id)
        results += run_tools(building_id, tools, ds(building_id), options)

    # Return the results
    return results


def run_tools(
    building_id: int, tools: list[Tool], data: Data, options: RunSettings
) -> list[Report]:
    """Run the benchmarks for the given tool."""

    # Run the benchmark for each tool
    results = []
    for tool in tools:
        logger.info("Running benchmarks for tool: %s", tool)
        results += run_methods(building_id, tool, data, options)

    # Return the results
    return results


def run_methods(
    building_id: int, tool: Tool, data: Data, options: RunSettings
) -> list[Report]:
    """Run benchmark methods with the given parameters."""

    # Prepare benchmark for tool and data
    logger.info("Preparing benchmark for tool: %s", tool)
    benchmark = Benchmark(building_id, tool, data, options)

    # Run benchmark for each tool
    results = []
    for method in tool.methods:
        logger.info("Running benchmark for method: %s", method)
        results.append(benchmark(method))

    # Return results
    return results


# Run main function if the script is executed
if __name__ == "__main__":
    main(options=Settings())
