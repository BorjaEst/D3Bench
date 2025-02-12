from frouros.detectors import concept_drift, data_drift

from d3bench.utils import BaseTestMethod, Result


class KSWIN(BaseTestMethod):
    """Kolmogorov-Smirnov Windowing detector."""

    def __init__(self):
        config = concept_drift.KSWINConfig(seed=31)
        self.detector = concept_drift.KSWIN(config)
        self._drifts = []

    def fit(self, x_reference: list[int | float]) -> None:
        # Detector is trained one by one on the reference data
        for x in x_reference:
            self.detector.update(value=x)

    def test(self, x_test: list[int | float]) -> None:
        for x in x_test:
            self.detector.update(value=x)
            self._drifts.append(self.detector.status["drift"])

    def result(self) -> Result:
        return Result(
            drift_detected=any(self._drifts),
        )


class CVMTest(BaseTestMethod):
    """Cramer-von Mises test for data drift detection."""

    def __init__(self):
        self.detector = data_drift.CVMTest()
        self._drift = None

    def fit(self, x_reference: list[int | float]) -> None:
        self.detector.fit(X=x_reference)

    def test(self, x_test: list[int | float]) -> None:
        self._drift = self.detector.compare(X=x_test)[0]

    def result(self) -> Result:
        return Result(
            p_value=self._drift.p_value,
            statistic=self._drift.statistic,
        )
