"""
This module provides a command-line interface to run D3Bench benchmarks.

The script allows users to specify various parameters for the benchmark,
including the buildings to benchmark, the criteria to test, the tools to use,
whether to run on a VM, the dataset to use, and the logging level.
"""

import logging

from d3bench.benchmark import Benchmark
from d3bench.config import RunSettings, Settings
from d3bench.dataset import Data_Energy, Data_Occupancy, Dataset
from d3bench.tool import AlibiDetect, Evidently, NannyML, Tool

logger = logging.getLogger(__name__)

DATASETS = {
    "energy": Data_Energy(),
    "occupancy": Data_Occupancy(),
}

TOOLS = {
    "Evidently": Evidently(),
    "NannyML": NannyML(),
    "Alibi-Detect": AlibiDetect(),
}


def main(options: Settings):
    """Run the benchmark with the given arguments."""

    # Print benchmark start message
    print("---------Benchmark execution started---------")

    # Set the logging level from the arguments
    logger.setLevel(options.log_level)
    logger.debug("args: %s", options)

    # Load dataset and tools from the arguments
    dataset = DATASETS[options.dataset]
    tools = [TOOLS[tool] for tool in options.tools]

    # Run the benchmark with the given parameters
    for tool in tools:
        logger.info("Running benchmark for tool: %s", tool)
        run_benchmark(tool, dataset, options)

    # Print the benchmark end message
    print("---------Benchmark execution completed---------")


def run_benchmark(tool: Tool, dataset: Dataset, options: RunSettings):
    """Run benchmark with the given parameters."""

    # Run benchmark for each tool
    for method in tool.methods:
        logger.info("Running benchmark for method: %s", method)
        benchmark = Benchmark(tool, method, dataset, options)
        benchmark.run()


# Run main function if the script is executed
if __name__ == "__main__":
    main(options=Settings())
