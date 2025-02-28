"""
This module provides a command-line interface to run D3Bench benchmarks.

The script allows users to specify various parameters for the benchmark,
including the buildings to benchmark, the criteria to test, the tools to use,
whether to run on a VM, the dataset to use, and the logging level.
"""

import datetime as dt
import logging
from typing import Literal, TypeAlias

from pydantic import Field
from pydantic_settings import SettingsConfigDict
from rich import print  # pylint: disable=redefined-builtin
from rich.logging import RichHandler

import d3bench
from d3bench import benchmarks, datasets, tools
from d3bench.config import Datafile, Framework

# pylint: disable=too-few-public-methods


LogLevel: TypeAlias = Literal["debug", "info", "warning", "error", "critical"]
logger = logging.getLogger(__name__)


class RunSettings(benchmarks.Options, datasets.Options, tools.Options):
    """Settings to run a benchmark."""


class Arguments(RunSettings, d3bench.ResultsOptions):
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
        default="info",
        description="Logging level.",
    )
    tools: set[Framework] = Field(
        # default=set(["Frouros", "Evidently", "NannyML", "Alibi-Detect"]),
        default=set(["Frouros", "Evidently", "NannyML"]),
        description="List of tools to benchmark.",
    )
    datafile: Datafile = Field(
        default="energy",
        description="Dataset file name to use.",
    )
    output: str = Field(
        default=f"results{dt.datetime.now().strftime('%Y%m%d%H%M%S')}",
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
    data = d3bench.DATASETS[args.datafile].split_data()
    _tools = [d3bench.TOOLS[tool](data, args) for tool in args.tools]

    # Run the benchmark with the given parameters
    print("------ Benchmark execution in progress ------")
    results = [d3bench.Results(tool, args) for tool in _tools]

    # Save the results to a file
    d3bench.save_results(results, args.output)

    # Print the benchmark end message
    print("---------Benchmark execution completed-------")


# Run main function if the script is executed
if __name__ == "__main__":
    main(args=Arguments())
