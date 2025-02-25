"""
This module provides a command-line interface to run D3Bench benchmarks.

The script allows users to specify various parameters for the benchmark,
including the buildings to benchmark, the criteria to test, the tools to use,
whether to run on a VM, the dataset to use, and the logging level.
"""

import logging
from typing import Literal, TypeAlias

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from rich import print  # pylint: disable=redefined-builtin
from rich.logging import RichHandler

from d3bench import DATASETS, TOOLS, utils
from d3bench.benchmark import BenchmarkOptions, run_tools
from d3bench.config import Dataset, Framework
from d3bench.dataset import DatasetOptions
from d3bench.tools import ToolOptions

# pylint: disable=too-few-public-methods


LogLevel: TypeAlias = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
logger = logging.getLogger(__name__)


class RunSettings(BenchmarkOptions, DatasetOptions, ToolOptions):
    """Settings to run a benchmark."""


class Arguments(RunSettings, BaseSettings):
    """
    This module provides a command-line interface to run D3Bench benchmarks.

    The script allows users to specify various parameters for the benchmark,
    including the buildings to benchmark, the criteria to test, the tools to
    use, whether to run on a VM, the dataset to use, and the logging level.
    """  # Description for the script help message

    # Class attributes
    model_config = SettingsConfigDict(
        cli_prog_name="python -m d3bench",
        cli_parse_args=True,
    )

    # Logging and reporting
    log_level: LogLevel = Field(
        default="INFO",
        description="Logging level.",
    )
    tools: set[Framework] = Field(
        default=set(["Frouros", "Evidently", "NannyML", "Alibi-Detect"]),
        description="List of tools to benchmark.",
    )
    dataset: Dataset = Field(
        default="energy",
        description="Dataset to use.",
    )
    results_file: str = Field(
        default="results.json",
        description="File to save the results to.",
    )


def main(args: Arguments) -> None:
    """Run the benchmark with the given arguments."""

    # Run the benchmark with the given parameters
    print("------ Benchmark script started -------------")

    # Set the logging level from the arguments
    logging.basicConfig(
        handlers=[RichHandler(rich_tracebacks=True)],
        level=args.log_level,
    )

    # Load dataset and tools from the arguments
    logger.debug("Call arguments: %s", args)
    data = DATASETS[args.dataset].split_data()
    tools = [TOOLS[tool] for tool in args.tools]

    # Run the benchmark with the given parameters
    print("------ Benchmark execution in progress ------")
    results = run_tools(tools, data, args)

    # Print the results to the console
    print(results)

    # Save the results to a file
    utils.save_results(results, args.results_file)

    # Print the benchmark end message
    print("---------Benchmark execution completed-------")


# Run main function if the script is executed
if __name__ == "__main__":
    main(args=Arguments())
