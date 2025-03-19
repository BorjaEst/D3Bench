"""Module for Frouros detectors."""

from typing import Any

from abc import ABC, abstractmethod
import pandas as pd
from frouros.detectors import concept_drift, data_drift
import numpy as np

from d3bench import utils
from frouros.utils.kernels import rbf_kernel

# TODO: Might be interesting to move all configurations to a toml file


# Online Concept Drift Detection


class BaseOnlineCD(utils.BaseTestMethod, ABC):
    """Base class for online concept drift detectors."""

    def __init__(self, features: list[str]) -> None:
        self.detectors = [self.detector_class(self.config) for _ in features]
        self.features = features
        self.drift: list[bool] = []

    @property
    @abstractmethod
    def config(self) -> Any:
        """Property that returns the detector configuration."""

    @property
    @abstractmethod
    def detector_class(self) -> Any:
        """Property that returns the detector class."""

    def fit(self, x_reference: np.ndarray) -> None:
        # Detector is trained one by one on the reference data
        # See:
        # https://frouros.readthedocs.io/en/latest/examples/concept_drift/DDM_advance.html#warm-up-phase
        for i, _ in enumerate(self.features):
            # !!! only 1000 instances are used for training, very high time consumption
            for x in x_reference[i][:1000]:
                self.detectors[i].update(value=x)

    def test(self, x_test: np.ndarray) -> None:
        # Only one feature is accepted
        for i, _ in enumerate(self.features):
            # !!! only 100 instances are used for testing, very high time consumption
            for x in x_test[:100, i]:
                self.detectors[i].update(value=x)

    def result(self) -> dict[str, Any]:
        return {"drifts": [d.status["drift"] for d in self.detectors]}


class BOCD(BaseOnlineCD):
    """Bayesian Online Change Detection."""

    detector_class = concept_drift.BOCD
    config = concept_drift.BOCDConfig(
        model=None,
        hazard=1e-2,  # hazard rate
        min_num_instances=1000,  # instances to start looking changes
    )


class CUSUM(BaseOnlineCD):
    """Cumulative Sum Control Chart"""

    detector_class = concept_drift.CUSUM
    config = concept_drift.CUSUMConfig(
        delta=1e-3,  # Delta property
        lambda_=50,  # Threshold property
        min_num_instances=1000,  # instances to start looking changes
    )


class GMA(BaseOnlineCD):
    """Geometric Moving Average"""

    detector_class = concept_drift.GeometricMovingAverage
    config = concept_drift.GeometricMovingAverageConfig(
        alpha=0.99,  # Forgetting factor value
        lambda_=1.0,  # Threshold property
        min_num_instances=1000,  # instances to start looking changes
    )


class PHT(BaseOnlineCD):
    """Page-Hinkley Test"""

    detector_class = concept_drift.PageHinkley
    config = concept_drift.PageHinkleyConfig(
        delta=1e-3,  # Delta property
        lambda_=50,  # Threshold property
        alpha=0.9999,  # Forgetting factor value
        min_num_instances=1000,  # instances to start looking changes
    )


class DDM(BaseOnlineCD):
    """Drift Detection Method"""

    detector_class = concept_drift.DDM
    config = concept_drift.DDMConfig(
        warning_level=2.0,  # Warning level property
        drift_level=3.0,  # Drift level property
        min_num_instances=1000,  # instances to start looking changes
    )


class ECDDWT(BaseOnlineCD):
    """EWMA Concept Drift Detection Warning"""

    detector_class = concept_drift.ECDDWT
    config = concept_drift.ECDDWTConfig(
        lambda_=0.2,  # Weight given to recent data compared to older data
        average_run_length=400,  # Expected time between false positive detections
        warning_level=0.5,  # Warning level property
        min_num_instances=1000,  # instances to start looking changes
    )


class EDDM(BaseOnlineCD):
    """Early Drift Detection Method"""

    detector_class = concept_drift.EDDM
    config = concept_drift.EDDMConfig(
        alpha=0.95,  # Warning zone value
        beta=0.9,  # Change zone value
        level=2.0,  # Drift level factor
        min_num_misclassified_instances=1000,  # instances to start looking changes
    )


class HDDM_A(BaseOnlineCD):
    """Hoeffding's Drift Detection Method Test-A"""

    detector_class = concept_drift.HDDMA
    config = concept_drift.HDDMAConfig(
        alpha_d=0.001,  # Significance level for the drift
        alpha_w=0.005,  # Significance level for the warning
        two_sided_test=False,  # Two-sided test flag
        min_num_instances=1000,  # instances to start looking changes
    )


class HDDM_W(BaseOnlineCD):
    """Hoeffding's Drift Detection Method Test-W"""

    detector_class = concept_drift.HDDMW
    config = concept_drift.HDDMWConfig(
        alpha_d=0.001,  # Significance level for the drift
        alpha_w=0.005,  # Significance level for the warning
        two_sided_test=False,  # Two-sided test flag
        lambda_=0.05,  # weight given to recent data compared to older data
        min_num_instances=1000,  # instances to start looking changes
    )


class RDDM(BaseOnlineCD):
    """Reactive Drift Detection Method"""

    detector_class = concept_drift.RDDM
    config = concept_drift.RDDMConfig(
        warning_level=1.773,  # Warning level property
        drift_level=2.258,  # Drift level property
        max_concept_size=40000,  # Maximum number of instances to consider
        max_num_instances_warning=7000,  # Maximum number of instances warning level
        min_num_instances=100,  # instances to start looking changes
    )


class ADWIN(BaseOnlineCD):
    """Adaptive Windowing"""

    detector_class = concept_drift.ADWIN
    config = concept_drift.ADWINConfig(
        clock=32,  # Clock property
        delta=0.002,  # Confidence value
        m=5,  # Amount of mem and closeness of cutpoints checked
        min_window_size=5,  # Minimum number of instances per window to start looking changes
        min_num_instances=1000,  # instances to start looking changes
    )


class KSWIN(BaseOnlineCD):
    """Kolmogorov-Smirnov Windowing detector."""

    detector_class = concept_drift.KSWIN
    config = concept_drift.KSWINConfig(
        alpha=1e-4,  # significance level
        seed=31,  # random seed
        min_num_instances=1000,  # instances to start looking changes
        num_test_instances=30,  # instances used by statistical test
    )


class STEPD(BaseOnlineCD):
    """Statistical Test of Equal Proportions"""

    detector_class = concept_drift.STEPD
    config = concept_drift.STEPDConfig(
        alpha_d=0.003,  # Significance value for overall
        alpha_w=0.005,  # Significance value for last
        min_num_instances=1000,  # instances to start looking changes
    )


# Online Data Drift Detection


class MMD(utils.BaseTestMethod):
    """Maximum Mean Discrepancy"""

    detector_class = data_drift.MMDStreaming
    config = {
        "window_size": 10,  # Window size value
        "kernel": rbf_kernel,  # Kernel function
        "chunk_size": 1000,  # Chunk size value
        "callbacks": None,  # Callbacks
    }

    def __init__(self, features: list[str]) -> None:
        self.detector = self.detector_class(**self.config)
        self.features = features
        self.distance = None

    def fit(self, x_reference: np.ndarray) -> None:
        self.detector.fit(X=x_reference)

    def test(self, x_test: np.ndarray) -> None:
        # Take only the first 10 instances for testing as window size is 10
        self.distance = [self.detector.update(value=x)[0] for x in x_test[:10]][-1]

    def result(self) -> dict[str, Any]:
        return {"distance": self.distance}


class KSI(utils.BaseTestMethod):
    """Incremental Kolmogorov-Smirnov Test"""

    detector_class = data_drift.IncrementalKSTest
    config = {
        "window_size": 10,  # Window size value
        "callbacks": None,  # Callbacks
    }

    def __init__(self, features: list[str]) -> None:
        self.detectors = [self.detector_class(**self.config) for _ in features]
        self.features = features
        self.results: list[Any] = [None for _ in features]

    def fit(self, x_reference: np.ndarray) -> None:
        # Only one feature is accepted
        for i, _ in enumerate(self.features):
            # 1000 values otherwise "IndexError": invalid index to scalar variable.
            self.detectors[i].fit(X=x_reference[:1000, i])

    def test(self, x_test: np.ndarray) -> None:
        # Only one feature is accepted
        for i, _ in enumerate(self.features):
            res = [self.detectors[i].update(value=x)[0] for x in x_test[:10, i]][-1]
            self.results[i] = res

    def result(self) -> dict[str, Any]:
        return {
            "distances": [self.results[i].statistic for i, _ in enumerate(self.features)],
            "p_values": [self.results[i].p_value for i, _ in enumerate(self.features)],
        }


# Batch Concept Drift Detection


# Batch Data Drift Detection


class BaseBatchDD(utils.BaseTestMethod, ABC):
    """Base class for batch data drift detectors."""

    def __init__(self, features: list[str]) -> None:
        self.detectors = [self.detector_class(**self.config) for _ in features]
        self.features = features
        self.results: list[Any] = []

    @property
    @abstractmethod
    def config(self) -> Any:
        """Property that returns the detector configuration."""

    @property
    @abstractmethod
    def detector_class(self) -> Any:
        """Property that returns the detector class."""

    def fit(self, x_reference: np.ndarray) -> None:
        for i, _ in enumerate(self.features):
            self.detectors[i].fit(X=x_reference[i])

    def test(self, x_test: np.ndarray) -> None:
        self.results = [
            self.detectors[i].compare(X=x_test[i])
            for i, _ in enumerate(self.features)
        ] # fmt: skip

    def result(self) -> dict[str, Any]:
        return {"results": self.results}


class BHATTACHARYYA(BaseBatchDD):
    """Bhattacharyya Distance"""

    detector_class = data_drift.BhattacharyyaDistance
    config = {"callbacks": None}


class KSTest(BaseBatchDD):
    """Kolmogorov-Smirnov Test"""

    detector_class = data_drift.KSTest
    config = {
        "num_bins": 10,  # number of bins in which to divide probabilities
        "callbacks": None,
    }
