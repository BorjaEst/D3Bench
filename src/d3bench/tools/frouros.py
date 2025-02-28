"""Module for Frouros detectors."""

from typing import Any

import pandas as pd
from frouros.detectors import concept_drift, data_drift

from d3bench import utils


# Online Supervised Concept Drift Detection


class KSWIN(utils.BaseTestMethod):
    """Kolmogorov-Smirnov Windowing detector."""

    def __init__(self, features: list[str]) -> None:
        config = concept_drift.KSWINConfig(
            alpha=1e-4,  # significance level
            seed=31,  # random seed
            min_num_instances=100,  # instances to start looking changes
            num_test_instances=30,  # instances used by statistical test
        )
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
        return self.detector.to_dict()


# Online Unsupervised Data Drift Detection

# Batch Concept Drift Detection


# Batch Data Drift Detection


class KSTest(utils.BaseTestMethod):
    """Kolmogorov-Smirnov Test"""

    def __init__(self, features: list[str]) -> None:
        self.detectors = {k: data_drift.KSTest() for k in features}
        self.features = features
        self._results: dict[str, Any] = {}

    def fit(self, x_reference: pd.DataFrame) -> None:
        for i, feature in enumerate(self.features):
            self.detectors[feature].fit(X=x_reference[i])

    def test(self, x_test: pd.DataFrame) -> None:
        self._results = {
            feature: self.detectors[feature].compare(X=x_test[i])[0]
            for i, feature in enumerate(self.features)
        }

    def result(self) -> dict[str, Any]:
        return self.detectors.to_dict()


class CVMTest(utils.BaseTestMethod):
    """Cramer-von Mises test for data drift detection."""

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
        return self.detectors.to_dict()
