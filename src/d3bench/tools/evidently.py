"""Module for Evidently detectors."""

from typing import Any

import pandas as pd
from evidently import ColumnMapping
from evidently.test_suite import TestSuite
from evidently.tests import TestColumnDrift

from d3bench import utils


# Online Supervised Concept Drift Detection


# Online Unsupervised Data Drift Detection


# Batch Concept Drift Detection


# Batch Data Drift Detection


class KSTest(utils.BaseTestMethod):
    """Kolmogorov-Smirnov Test"""

    def __init__(self, features: list[str]) -> None:
        tests = [TestColumnDrift(feature) for feature in features]
        self.test_suite = TestSuite(tests)
        self.kwds = {
            "column_mapping": ColumnMapping(),
            "reference_data": None,
        }

    def fit(self, x_reference: pd.DataFrame) -> None:
        # No train method separated from the test method in Evidetly
        self.kwds["reference_data"] = x_reference
        raise NotImplementedError(f"No train for {self.__class__}")

    def test(self, x_test: pd.DataFrame) -> None:
        self.test_suite.run(current_data=x_test, **self.kwds)

    def result(self) -> dict[str, Any]:
        return self.test_suite.as_dict()
