"""Utility functions and classes for drift detection benchmark results.

This module provides the data classes needed to store, analyze, and compare
benchmark results across different drift detection methods and frameworks.
"""

import dataclasses as dc
import datetime as dt
from typing import Any, Dict, List, Optional, Union

import numpy as np

from d3bench import methods
from d3bench.benchmarks import Benchmark
from d3bench.config import Criteria, Framework

# pylint: disable=too-many-instance-attributes
# pylint: disable=too-few-public-methods


@dc.dataclass(init=False)
class TestInformation:
    """Information about the test method used in the benchmark.

    This class is used to store information about the test method used in the
    benchmark. It includes the framework used, whether the benchmark was run
    on a VM, the number of repetitions, and the length of the training and
    test data.
    """

    framework: Framework  # tool used in the benchmark
    run_on_vm: bool  # run on a VM
    repetitions: int  # number of repetitions
    len_reference: int  # length of the training data
    len_testing: int  # length of the test data

    def __init__(self, benchmark: Benchmark) -> None:
        self.framework = benchmark.tool.name
        self.run_on_vm = benchmark.run_on_vm
        self.repetitions = benchmark.repetitions
        self.len_reference = benchmark.data.len_reference
        self.len_testing = benchmark.data.len_testing


@dc.dataclass(init=False)
class Stats:
    """Statistics computed from multiple benchmark runs.

    Note: It's tempting to calculate mean and standard deviation from the
    result vector and report these. However, this is not very useful. In
    a typical case, the lowest value gives a lower bound for how fast your
    machine can run the given code snippet; higher values in the result
    vector are typically not caused by variability in Python's speed, but
    by other processes interfering with your timing accuracy. So the min()
    of the result is probably the only number you should be interested in.

    After that, you should look at the entire vector and apply common sense
    rather than statistics.
    """

    avg: float  # Average value
    max: float  # Maximum value
    min: float  # Minimum value

    def __init__(self, values: list[float | int]) -> None:
        values_array = np.array(values, dtype=float)
        self.avg = float(values_array.mean())
        self.max = float(values_array.max())
        self.min = float(values_array.min())


@dc.dataclass(init=False)
class Report:
    """
    Base data class to store the results of the benchmark.
    This is the parent class for all specific report types.
    """

    test_information: TestInformation  # info about test method
    time: dt.datetime  # time of the benchmark
    runtime: Optional[Stats] = None  # runtime statistics
    cputime: Optional[Stats] = None  # runtime statistics
    memory: Optional[Stats] = None  # runtime statistics

    def __init__(self, criteria: set[Criteria], benchmark: Benchmark) -> None:
        self.test_information = TestInformation(benchmark)
        self.time = dt.datetime.now()
        if "runtime" in criteria:
            self.runtime = Stats(benchmark.get_runtimes())
        if "cputime" in criteria:
            self.cputime = Stats(benchmark.get_cputimes())
        if "memory" in criteria:
            self.memory = Stats(benchmark.get_memories())


@dc.dataclass(init=False)
class OnlineCDReport(Report):
    """
    Data class to store the results of a Online Supervised Concept Drift
    Detection benchmark.

    This report captures metrics specific to online supervised drift detection
    methods, focusing on detection speed, accuracy and false
    positives/negatives.
    """

    method: methods.OnlineCD  # method used in benchmark
    detection_delay: Optional[Stats] = None  # samples until drift detection
    false_alarm_rate: Optional[float] = None  # false positives
    missed_detection_rate: Optional[float] = None  # false negatives
    f1_score: Optional[float] = None  # harmonic mean of precision and recall
    adaptation_time: Optional[Stats] = None  # time to adapt after drift
    auc_score: Optional[float] = None  # area under ROC

    def __init__(self, criteria: set[Criteria], benchmark: Benchmark) -> None:
        super().__init__(criteria=criteria, benchmark=benchmark)
        if not isinstance(benchmark.method, methods.OnlineCD):
            raise ValueError("Method must be an OnlineCD instance")
        self.method = benchmark.method


@dc.dataclass(init=False)
class OnlineDDReport(Report):
    """
    Data class to store the results of a Online Unsupervised Data Drift
    Detection benchmark.

    This report captures metrics specific to online unsupervised drift
    detection, focusing on statistical properties and computational
    efficiency.
    """

    method: methods.OnlineDD  # method used in benchmark
    detection_delay: Optional[Stats] = None  # samples until drift detection
    false_alarm_rate: Optional[float] = None  # false positives
    statistical_power: Optional[float] = None  # ability to detect drift
    processing_time_per_sample: Optional[Stats] = None  # comp. efficiency
    test_statistics: Optional[Dict[str, Any]] = None  # method-specific stats

    def __init__(self, criteria: set[Criteria], benchmark: Benchmark) -> None:
        super().__init__(criteria=criteria, benchmark=benchmark)
        if not isinstance(benchmark.method, methods.OnlineDD):
            raise ValueError("Method must be an OnlineDD instance")
        self.method = benchmark.method
        # TODO: add the rest of the metrics


@dc.dataclass(init=False)
class BatchCDReport(Report):
    """
    Data class to store the results of a Batch Concept Drift Detection
    benchmark.

    This report captures metrics specific to batch concept drift methods,
    focusing on detection accuracy and characteristics of identified drift.
    """

    method: methods.BatchCD  # method used in benchmark
    drift_detected: bool = False  # whether drift was detected
    detection_accuracy: Optional[float] = None  # correct detections
    drift_magnitude: Optional[float] = None  # magnitude of the drift
    confidence_level: Optional[float] = None  # confidence in detection
    drift_location: Optional[Union[int, List[int]]] = None  # est. drift pos.
    testing_power: Optional[float] = None  # statistical power

    def __init__(self, criteria: set[Criteria], benchmark: Benchmark) -> None:
        super().__init__(criteria=criteria, benchmark=benchmark)
        if not isinstance(benchmark.method, methods.BatchCD):
            raise ValueError("Method must be a BatchCD instance")
        self.method = benchmark.method
        # TODO: add the rest of the metrics


@dc.dataclass(init=False)
class BatchDDReport(Report):
    """
    Data class to store the results of a Batch Data Drift Detection
    benchmark.

    This report captures metrics specific to batch data drift detection,
    focusing on statistical measures and feature-level drift analysis.
    """

    method: methods.BatchDD  # method used in benchmark
    drift_detected: bool = False  # whether drift was detected
    drift_score: Optional[float] = None  # overall magnitude
    feature_drift_scores: Optional[Dict[str, float]] = None  # per-feature
    p_value: Optional[float] = None  # significance
    effect_size: Optional[float] = None  # magnitude of the effect
    confidence_interval: Optional[tuple] = None  # confidence interval

    def __init__(self, criteria: set[Criteria], benchmark: Benchmark) -> None:
        super().__init__(criteria=criteria, benchmark=benchmark)
        if not isinstance(benchmark.method, methods.BatchDD):
            raise ValueError("Method must be a BatchDD instance")
        self.method = benchmark.method
        # TODO: add the rest of the metrics
