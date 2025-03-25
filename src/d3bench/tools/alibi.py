"""Module for Alibi Detect detectors."""

from typing import Any

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from alibi_detect import cd

from d3bench import utils


# Concept Drift Univariate Detector Methods
class BaseUnivariateTest(utils.BaseTestMethod, ABC):
    """
    Base class for univariate drift detectors.

    drift_type
        Predict drift at the 'feature' or 'batch' level. For 'batch', the test statistics for
        each feature are aggregated using the Bonferroni or False Discovery Rate correction (if n_features>1).
    """

    def __init__(self, features: list[str]) -> None:
        self.features = features
        self.detector: Any
        self.drift: Any

    @property
    @abstractmethod
    def config(self) -> Any:
        """Property that returns the detector configuration."""

    @property
    @abstractmethod
    def detector_class(self) -> Any:
        """Property that returns the detector class."""

    def fit(self, x_reference: np.ndarray) -> None:
        self.detector = self.detector_class(x_reference, **self.config)

    def test(self, x_test: np.ndarray) -> None:
        self.drift = self.detector.predict(x_test, drift_type="feature")

    def result(self) -> dict[str, Any]:
        return NotImplementedError("Method not implemented.")


class KolmogorovSmirnovTest(BaseUnivariateTest):
    """Kolmogorov-Smirnov test for univariate drif detection.
    TODO: !Is this Batch CD test?!
    """

    detector_class = cd.KSDrift
    config = {
        "p_val": 0.05,
        "x_ref_preprocessed": False,
        "preprocess_at_init": True,
        "update_x_ref": None,
        "preprocess_fn": None,
        "correction": "bonferroni",
        "alternative": "two-sided",
        "n_features": None,
        "input_shape": None,
        "data_type": None,
    }
