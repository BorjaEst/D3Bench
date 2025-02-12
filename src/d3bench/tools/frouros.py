from frouros.detectors import concept_drift, data_drift

from d3bench.utils import BaseTestMethod, Result


class KSWIN(BaseTestMethod):
    """Kolmogorov-Smirnov Windowing detector."""

    def __init__(self):
        self.detector = concept_drift.KSWIN(
            config=concept_drift.KSWINConfig(seed=31),
            callbacks=None,
        )
        self._values = []

    def fit(self, x_reference: list[int | float]) -> None:
        # Detector is trained one by one on the reference data
        for x in x_reference:
            self.detector.update(value=x)

    def test(self, x_test: list[int | float]) -> None:
        for x in x_test:
            self.detector.update(value=x)
            self._values.append(
                self.detector.status["p_value"],
                self.detector.drift,
            )

    def result(self) -> Result:
        p_value, drift = max(self._values, key=lambda x: x[0])
        return Result(p_value, drift)


class CVMTest(BaseTestMethod):
    """Cramer-von Mises test for data drift detection."""

    def __init__(self):
        self.detector = data_drift.CVMTest(
            callbacks=None,
        )

    def fit(self, x_reference: list[int | float]) -> None:
        self.detector.fit(X=x_reference)

    def test(self, x_test: list[int | float]) -> None:
        self.detector.compare(X=x_test)

    def result(self) -> Result:
        raise RuntimeError()
        return Result(
            p_values=self.detector.p_value,
            drift_detected=1,
            statistic=self.detector.statistic,
        )
