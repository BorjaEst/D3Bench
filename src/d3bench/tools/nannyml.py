"""Module for Nanny ML detectors."""

from abc import ABC, abstractmethod
from typing import Any

import nannyml as nml
import pandas as pd

from d3bench import utils


# Univariate Continuous Data Drift Detection


class BaseUnivariateContinuous(utils.BaseTestMethod, ABC):
    """Base class for batch data drift detectors."""

    def __init__(self, features: list[str]) -> None:
        self.detector = nml.UnivariateDriftCalculator(
            column_names=features,
            chunk_number=None,
            timestamp_column_name="time",
            continuous_methods=[self.detector_reference],
        )
        self.results: Any = None

    @property
    @abstractmethod
    def detector_reference(self) -> Any:
        """Property that returns the detector class."""

    def fit(self, x_reference: pd.DataFrame) -> None:
        self.detector.fit(x_reference)

    def test(self, x_test: pd.DataFrame) -> None:
        self.results = self.detector.calculate(x_test)

    def result(self) -> dict[str, Any]:
        raise NotImplementedError


class JensenShannonDivergenceDriftDetection(BaseUnivariateContinuous):
    """Jensen-Shannon Divergence Drift Detection"""

    detector_reference = "jensen_shannon"


class WassersteinDistance(BaseUnivariateContinuous):
    """Wasserstein Distance"""

    detector_reference = "wasserstein"


class HellingerDistance(BaseUnivariateContinuous):
    """Hellinger Distance"""

    detector_reference = "hellinger"


class KolmogorovSmirnovTest(BaseUnivariateContinuous):
    """Kolmogorov-Smirnov Test"""

    detector_reference = "kolmogorov_smirnov"


# Univariate Categorical Data Drift Detection


class BaseUnivariateCategorical(utils.BaseTestMethod, ABC):
    """Base class for batch data drift detectors."""

    def __init__(self, features: list[str]) -> None:
        self.detector = nml.UnivariateDriftCalculator(
            column_names=features,
            chunk_number=None,
            timestamp_column_name="time",
            categorical_methods=[self.detector_reference],
        )
        self.results: Any = None

    @property
    @abstractmethod
    def detector_reference(self) -> Any:
        """Property that returns the detector class."""

    def fit(self, x_reference: pd.DataFrame) -> None:
        self.detector.fit(x_reference)

    def test(self, x_test: pd.DataFrame) -> None:
        self.results = self.detector.calculate(x_test)

    def result(self) -> dict[str, Any]:
        raise NotImplementedError


# class JensenShannonDivergenceDriftDetection(BaseUnivariateCategorical):
#     """Jensen-Shannon Divergence Drift Detection"""

#     detector_reference = "jensen_shannon"


# class HellingerDistance(BaseUnivariateCategorical):
#     """Hellinger Distance"""

#     detector_reference = "hellinger"


class ChiSquareTest(BaseUnivariateCategorical):
    """Chi-Square Test"""

    detector_reference = "chi2"


class LInfinityDistance(BaseUnivariateCategorical):
    """L-Infinity Distance"""

    detector_reference = "l_infinity"
