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

    def __init__(self, n_features: int) -> None:
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
            "drift_detected": any(self._drifts),
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

    def __init__(self, n_features: int) -> None:
        self.detectors = [data_drift.CVMTest() for _ in range(n_features)]
        self._results: list[Any] = []

    def fit(self, x_reference: pd.DataFrame) -> None:
        for i, feature in enumerate(x_reference.columns):
            self.detectors[i].fit(X=x_reference[feature])

    def test(self, x_test: pd.DataFrame) -> None:
        self._results = [
            self.detectors[i].compare(X=x_test[feature])[0]
            for i, feature in enumerate(x_test.columns)
        ]

    def result(self) -> dict[str, Any]:
        return {
            "p_values": [result.p_value for result in self._results],
            "statistics": [result.statistic for result in self._results],
        }
