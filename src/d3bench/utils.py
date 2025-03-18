"""Utility functions and classes for the drift detection methods."""

import dataclasses as dc
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
from pydantic_settings import (
    BaseSettings,
    CliSettingsSource,
    PydanticBaseSettingsSource,
)
from rich_argparse import RichHelpFormatter

# pylint: disable=too-many-instance-attributes
# pylint: disable=too-few-public-methods


class BaseArguments(BaseSettings):
    """Base class for all scripts"""

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Enable CLI formatter_class to work properly."""
        return (
            CliSettingsSource(
                settings_cls,
                formatter_class=RichHelpFormatter,
                cli_parse_args=True,
            ),
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )


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


@dc.dataclass
class Data:  # pylint: disable=missing-class-docstring
    features: list[str]
    reference: pd.DataFrame
    testing: pd.DataFrame

    @property
    def len_reference(self) -> int:
        """Return the number of samples in the reference data."""
        return self.reference.shape[0]

    @property
    def len_testing(self) -> int:
        """Return the number of samples in the testing data."""
        return self.testing.shape[0]
