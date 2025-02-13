"""Module for Frouros detectors."""

from typing import Any

import pandas as pd
from frouros.detectors import concept_drift, data_drift

from d3bench import utils


class KSWIN(utils.BaseTestMethod):
    """Kolmogorov-Smirnov Windowing detector."""

    @property
    def info(self) -> utils.DetectorInformation:
        return utils.DetectorInformation(
            multi_features=True,
            fit_method=False,
            detector_type="Concept drift",
            operation_type="Streaming",
        )

    def __init__(self, features: list[str]) -> None:
        config = concept_drift.KSWINConfig(seed=31)
        self.detector = concept_drift.KSWIN(config)
        self._drifts: list[Any] = []

    def fit(self, x_reference: pd.DataFrame) -> None:
        # Detector is trained one by one on the reference data
        for x in x_reference:
            self.detector.update(value=x)

    def test(self, x_test: pd.DataFrame) -> None:
        for x in x_test:
            self.detector.update(value=x)
            self._drifts.append(self.detector.status["drift"])

    def result(self) -> dict[str, Any]:
        return {
            "dataset_drift": any(self._drifts),
        }


class CVMTest(utils.BaseTestMethod):
    """Cramer-von Mises test for data drift detection."""

    @property
    def info(self) -> utils.DetectorInformation:
        return utils.DetectorInformation(
            multi_features=False,
            fit_method=True,
            detector_type="Data drift",
            operation_type="Batch",
        )

    def __init__(self, features: list[str]) -> None:
        self.detectors = {k: data_drift.CVMTest() for k in features}
        self._results: dict[str, Any] = {}

    def fit(self, x_reference: pd.DataFrame) -> None:
        for feature in x_reference.columns:
            self.detectors[feature].fit(X=x_reference[feature])

    def test(self, x_test: pd.DataFrame) -> None:
        self._results = {
            k: self.detectors[k].compare(X=x_test[k])[0]
            for k in x_test.columns  # fmt: skip
        }

    def result(self) -> dict[str, Any]:
        return {
            "p_values": {
                column: result.p_value  # fmt: skip
                for column, result in self._results.items()
            },
            "statistics": {
                column: result.statistic  # fmt: skip
                for column, result in self._results.items()
            },
        }
