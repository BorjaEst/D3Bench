"""Utility functions and classes for the drift detection methods."""

from abc import ABC, abstractmethod
from typing import Any


# pylint: disable=too-many-instance-attributes
# pylint: disable=too-few-public-methods


class BaseTestMethod(ABC):
    """Base class for the test methods."""

    @abstractmethod
    def __init__(self, features: list[str]) -> None:
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
