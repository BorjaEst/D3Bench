"""Module for Frouros detectors."""

from typing import Any

import nannyml as nml
import pandas as pd

from d3bench import utils


class KSWIN(utils.BaseTestMethod):
    """Kolmogorov-Smirnov Windowing detector."""

    def __init__(self, features: list[str]) -> None:
        self.detector = nml.UnivariateDriftCalculator(
            column_names=features,
            chunk_number=1,
            timestamp_column_name="time",
            continuous_methods=["kolmogorov_smirnov"],
        )
        self.results: Any = None

    def fit(self, x_reference: pd.DataFrame) -> None:
        self.detector.fit(x_reference)

    def test(self, x_test: pd.DataFrame) -> None:
        self.results = self.detector.calculate(x_test)

    def result(self) -> dict[str, Any]:
        results: pd.DataFrame = self.results.to_df().iloc[-1]
        return {
            "drift_detected": any(
                results[column].values[-1]
                for column in self.detector.continuous_column_names
            ),
            "p_values": {
                column: results[column].values[1]
                for column in self.detector.continuous_column_names
            },
        }


class KSWIN(utils.BaseTestMethod):
    """Kolmogorov-Smirnov Windowing detector."""

    def __init__(self, features: list[str]) -> None:
        self.detector = nml.UnivariateDriftCalculator(
            column_names=features,
            chunk_number=1,
            timestamp_column_name="time",
            continuous_methods=["kolmogorov_smirnov"],
        )
        self.results: Any = None

    def fit(self, x_reference: pd.DataFrame) -> None:
        self.detector.fit(x_reference)

    def test(self, x_test: pd.DataFrame) -> None:
        self.results = self.detector.calculate(x_test)

    def result(self) -> dict[str, Any]:
        results: pd.DataFrame = self.results.to_df().iloc[-1]
        return {
            "drift_detected": any(
                results[column].values[-1]
                for column in self.detector.continuous_column_names
            ),
            "p_values": {
                column: results[column].values[1]
                for column in self.detector.continuous_column_names
            },
        }
