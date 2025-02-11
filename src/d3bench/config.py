"""Configuration settings for the d3bench package."""

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from d3bench.benchmark import BenchmarkOptions
from d3bench.dataset import DatasetOptions, Dataset
from d3bench.tool import ToolOptions, Framework

# pylint: disable=too-few-public-methods


LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class RunSettings(BenchmarkOptions, DatasetOptions, ToolOptions):
    """Settings to run a benchmark."""


class Settings(RunSettings, BaseSettings):
    """
    This module provides a command-line interface to run D3Bench benchmarks.

    The script allows users to specify various parameters for the benchmark,
    including the buildings to benchmark, the criteria to test, the tools to
    use, whether to run on a VM, the dataset to use, and the logging level.
    """  # Description for the script help message

    # Class attributes
    model_config = SettingsConfigDict(cli_parse_args=True)

    # Logging and reporting
    log_level: LogLevel = Field(
        default="INFO",
        description="Logging level.",
    )
    buildings: set[int] = Field(
        default={1},
        description="List of building IDs to benchmark.",
    )
    tools: set[Framework] = Field(
        # default=set(["Evidently", "NannyML", "Alibi-Detect"]),
        default=set(["Evidently", "NannyML"]),
        description="List of tools to benchmark.",
    )
    dataset: Dataset = Field(
        default="energy",
        description="Dataset to use.",
    )
    results_file: str = Field(
        default="results.csv",
        description="File to save the results to.",
    )
