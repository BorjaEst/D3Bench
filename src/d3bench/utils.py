"""Utility functions and classes for the drift detection methods."""

import json
from abc import ABC, abstractmethod
from typing import Any

from pydantic.json import pydantic_encoder

from d3bench import config
from d3bench.results import Report

# pylint: disable=too-many-instance-attributes
# pylint: disable=too-few-public-methods


class BaseTestMethod(ABC):
    """Base class for the test methods."""

    @abstractmethod
    def __init__(self, n_features: int) -> None:
        """Initialize the test method."""

    @abstractmethod
    def fit(self, x_reference: Any) -> None:
        """Run the test on the reference data."""

    @abstractmethod
    def test(self, x_test: Any) -> None:
        """Run the test on the test data."""

    @abstractmethod
    def result(self) -> dict[str, Any]:
        """Return the result of the test."""


def save_results(
    results: list[Report],
    file_name: str,
) -> None:
    """Save the results to a JSON parsed file."""
    output = config.results_path / file_name
    results_json = json.dumps(
        results,
        indent=4,
        default=pydantic_encoder,
    )

    # Save the results to a file in JSON format
    with open(output, "w", encoding="utf-8") as file:
        file.write(results_json)
