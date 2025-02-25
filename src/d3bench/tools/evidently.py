"""Module for Evidently detectors."""

from typing import Any

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently import report

from d3bench import utils


class KSWIN(utils.BaseTestMethod):
    """Kolmogorov-Smirnov Windowing detector."""

    @property
    def info(self) -> utils.DetectorInformation:
        return utils.DetectorInformation(
            multivariate_detector=True,
            fit_method=False,
            detector_type="Concept drift",
            operation_type="Batch",
        )

    def __init__(self, features: list[str]) -> None:
        metrics = [DataDriftPreset(stattest="ks")]
        self.report = report.Report(metrics)
        self._run_test: Any = None

    def fit(self, x_reference: pd.DataFrame) -> None:
        # No train method separated from the test method in Evidetly
        self._run_test = lambda x: self.report.run(
            reference_data=x_reference, current_data=x
        )
        raise NotImplementedError(f"No train for {self.__class__}")

    def test(self, x_test: pd.DataFrame) -> None:
        self._run_test(x_test)

    def result(self) -> dict[str, Any]:
        result = self.report.as_dict()["metrics"][1]["result"]
        return {
            "drift_detected": result["dataset_drift"],
            "p_values": {
                column: values["drift_score"]
                for column, values in result["drift_by_columns"].items()
            },
        }
