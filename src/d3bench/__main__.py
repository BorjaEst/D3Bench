"""
This module provides a command-line interface to run D3Bench benchmarks.

The script allows users to specify various parameters for the benchmark,
including the buildings to benchmark, the criteria to test, the tools to use,
whether to run on a VM, the dataset to use, and the logging level.
"""

import logging

from d3bench.benchmark import Benchmark
from d3bench.config import RunSettings, Settings
from d3bench.dataset import Data_Energy, Data_Occupancy
from d3bench.tool import AlibiDetect, Evidently, NannyML

logger = logging.getLogger(__name__)


def main(settings: Settings):
    """Run the benchmark with the given arguments."""

    # Print benchmark start message
    print("---------Benchmark execution started---------")

    # Set the logging level from the arguments
    logger.setLevel(settings.log_level)
    logger.debug("args: %s", settings)

    # Run the benchmark with the given parameters
    for tool in set(settings.tools):
        logger.info("Running benchmark for tool: %s", tool)
        run_benchmark(tool, settings)

    # Print the benchmark end message
    print("---------Benchmark execution completed---------")


# one benchmark execution with given criteria, tools and dataset
def run_benchmark(tool: str, settings: RunSettings):
    """Run benchmark with the given parameters."""

    # Convert the tool name to the corresponding tool object
    match tool:
        case "Evidently":
            tool = Evidently("Evidently", settings)
        case "NannyML":
            tool = NannyML("NannyML", settings)
        case "AlibiDetect":
            tool = AlibiDetect("AlibiDetect", settings)

    # Convert the dataset name to the corresponding dataset object
    match settings.dataset:
        case "energy":
            dataset = Data_Energy(settings)
        case "occupancy":
            dataset = Data_Occupancy(settings)

    # Run benchmark for each tool
    benchmark = Benchmark(tool, dataset, settings)
    benchmark.runBenchmark()


# Run main function if the script is executed
if __name__ == "__main__":
    main(Settings())
