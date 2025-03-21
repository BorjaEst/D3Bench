"""Module for Evidently detectors."""

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
from evidently.future.metrics import ValueDrift
from evidently.future.report import Report

from d3bench import utils

# Tabular Univariate Data Drift Detection


class BaseTabularDetectors(utils.BaseTestMethod, ABC):
    """Base class for Evidently tabular detectors."""

    def __init__(self, features: list[str]) -> None:
        self.reports = {f: Report([ValueDrift(column=f, method=self.detector_reference)])
                        for f in features} # fmt: skip
        self.__x_reference: pd.DataFrame
        self.results: dict[str, Any]

    @property
    @abstractmethod
    def detector_reference(self) -> Any:
        """Property that returns the detector class."""

    def fit(self, x_reference: pd.DataFrame) -> None:
        self.__x_reference = x_reference
        raise NotImplementedError("Evidence does not provide fit method")

    def test(self, x_test: pd.DataFrame) -> None:
        x_reference = self.__x_reference
        self.results = {f: self.reports[f].run(x_reference, x_test)
                        for f in self.reports} # fmt: skip

    def result(self) -> dict[str, Any]:
        return {
            "p-values": {f: self.results[f].dict()["metrics"][0]["value"] 
                         for f in self.results}, # fmt: skip
        }


class KolmogorovSmirnovTest(BaseTabularDetectors):
    """Kolmogorov-Smirnov Test"""

    detector_reference = "ks"


class ChiSquareTest(BaseTabularDetectors):
    """Chi-Square Test"""

    detector_reference = "chisquare"


class ZTest(BaseTabularDetectors):
    """Z Test"""

    detector_reference = "z"


class WassersteinDistance(BaseTabularDetectors):
    """Wasserstein Distance"""

    detector_reference = "wasserstein"


class KullbackLeiblerDivergenceDriftDetection(BaseTabularDetectors):
    """Kullback-Leibler Divergence Drift Detection"""

    detector_reference = "kl_div"


class PopulationStabilityIndex(BaseTabularDetectors):
    """Population Stability Index"""

    detector_reference = "psi"


class JensenShannonDivergenceDriftDetection(BaseTabularDetectors):
    """Jensen-Shannon Divergence Drift Detection"""

    detector_reference = "jensenshannon"


class AndersonDarlingTest(BaseTabularDetectors):
    """Anderson-Darling Test"""

    detector_reference = "anderson"


class FisherExactTest(BaseTabularDetectors):
    """Fisher Exact Test"""

    detector_reference = "fisher_exact"


class CramerVonMisesTest(BaseTabularDetectors):
    """Cramér-von Mises Test"""

    detector_reference = "cramer_von_mises"


class GTest(BaseTabularDetectors):
    """G Test"""

    detector_reference = "g-test"


class HellingerDistance(BaseTabularDetectors):
    """Hellinger Distance"""

    detector_reference = "hellinger"


class MannWhitneyUTest(BaseTabularDetectors):
    """Mann-Whitney U-Test"""

    detector_reference = "mannw"


class EnergyDistance(BaseTabularDetectors):
    """Energy Distance"""

    detector_reference = "ed"


class EppsSingletonTest(BaseTabularDetectors):
    """Epps-Singleton Test"""

    detector_reference = "es"


class TTest(BaseTabularDetectors):
    """T Test"""

    detector_reference = "t_test"


class EmpiricalMaximumMeanDiscrepancy(BaseTabularDetectors):
    """Empirical Maximum Mean Discrepancy"""

    detector_reference = "empirical_mmd"


class TotalVariationDistance(BaseTabularDetectors):
    """Total Variation Distance"""

    detector_reference = "TVD"
