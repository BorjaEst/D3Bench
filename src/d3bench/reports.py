"""Utility functions and classes for drift detection benchmark results.

This module provides the data classes needed to store, analyze, and compare
benchmark results across different drift detection methods and frameworks.
"""

import datetime as dt
import json
from typing import Any, Dict, List, Optional, Union

import numpy as np
from pydantic import BaseModel, Field, field_validator
from pydantic.json import pydantic_encoder

from d3bench import methods
from d3bench.benchmarks import Benchmark
from d3bench.config import Criteria, Framework

# pylint: disable=too-many-instance-attributes
# pylint: disable=too-few-public-methods


class TestInformation(BaseModel):
    """Information about the test method used in the benchmark.

    This class is used to store information about the test method used in the
    benchmark. It includes the framework used, whether the benchmark was run
    on a VM, the number of repetitions, and the length of the training and
    test data.
    """

    framework: Framework  # tool used in the benchmark
    run_on_vm: bool  # run on a VM
    repetitions: int  # number of repetitions
    len_traindata: int  # length of the training data
    len_testdata: int  # length of the test data

    @field_validator("repetitions", "len_traindata", "len_testdata")
    @classmethod
    def validate_positive_integers(cls, v: Any):
        """Validate that integers are positive."""
        if v <= 0:
            raise ValueError(f"{v} must be positive")
        return v

    def __init__(self, benchmark: Benchmark) -> None:
        super().__init__(
            framework=benchmark.tool.name,
            run_on_vm=benchmark.run_on_vm,
            repetitions=benchmark.repetitions,
            len_traindata=benchmark.x_reference.shape[0],
            len_testdata=benchmark.x_test.shape[0],
        )


class Stats(BaseModel):
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

    @classmethod
    def from_values(cls, values: list[float | int]) -> "Stats":
        """Create a Stats object from a list of values."""
        values_array = np.array(values, dtype=float)
        return cls(
            avg=float(values_array.mean()),
            max=float(values_array.max()),
            min=float(values_array.min()),
        )


class Report(BaseModel):
    """
    Base data class to store the results of the benchmark.
    This is the parent class for all specific report types.
    """

    time: dt.datetime = Field(default_factory=dt.datetime.now)  # time
    test_information: TestInformation  # information about the test method
    runtime: Optional[Stats] = None  # runtime statistics
    cputime: Optional[Stats] = None  # runtime statistics
    memory: Optional[Stats] = None  # runtime statistics

    class Config:  # pylint: disable=missing-class-docstring
        json_encoders = {dt.datetime: lambda v: v.isoformat()}

    def __init__(self, criteria: set[Criteria], benchmark: Benchmark) -> None:
        stats: dict[Criteria, Any] = {}  # dictionary to store statistics
        if "runtime" in criteria:
            stats["runtime"] = Stats.from_values(benchmark.get_runtimes())
        if "cputime" in criteria:
            stats["cputime"] = Stats.from_values(benchmark.get_cputimes())
        if "memory" in criteria:
            stats["memory"] = Stats.from_values(benchmark.get_memories())
        super().__init__(
            test_information=TestInformation(benchmark),
            runtime=stats.get("runtime", None),
            cputime=stats.get("cputime", None),
            memory=stats.get("memory", None),
        )

    def to_json(self) -> str:
        """Serialize report to JSON."""
        return json.dumps(self, indent=4, default=pydantic_encoder)


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
    false_alarm_rate: Optional[float] = Field(None, ge=0.0, le=1.0)
    missed_detection_rate: Optional[float] = Field(None, ge=0.0, le=1.0)
    f1_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    adaptation_time: Optional[Stats] = None  # time to adapt after drift
    auc_score: Optional[float] = Field(None, ge=0.0, le=1.0)  # area under ROC

    def __init__(self, criteria: set[Criteria], benchmark: Benchmark) -> None:
        super().__init__(criteria, benchmark)
        if not isinstance(benchmark.method, methods.OnlineCD):
            raise ValueError("Method must be an OnlineCD instance")
        self.method = benchmark.method
        # TODO: add the rest of the metrics

    @classmethod
    def detect_report_type(cls, data: Dict) -> bool:
        """Detect if the data represents this report type."""
        if not isinstance(data.get("method"), dict):
            return False
        method = data.get("method", {}).get("type", "")
        return "OnlineSupervisedConceptDrift" in str(method)


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
    false_alarm_rate: Optional[float] = Field(None, ge=0.0, le=1.0)
    statistical_power: Optional[float] = Field(None, ge=0.0, le=1.0)
    processing_time_per_sample: Optional[Stats] = None  # comp. efficiency
    test_statistics: Optional[Dict[str, Any]] = None  # method-specific stats

    def __init__(self, criteria: set[Criteria], benchmark: Benchmark) -> None:
        super().__init__(criteria, benchmark)
        if not isinstance(benchmark.method, methods.OnlineDD):
            raise ValueError("Method must be an OnlineDD instance")
        self.method = benchmark.method
        # TODO: add the rest of the metrics

    @classmethod
    def detect_report_type(cls, data: Dict) -> bool:
        """Detect if the data represents this report type."""
        if not isinstance(data.get("method"), dict):
            return False
        method = data.get("method", {}).get("type", "")
        return "OnlineUnsupervisedDataDrift" in str(method)


class BatchCDReport(Report):
    """
    Data class to store the results of a Batch Concept Drift Detection
    benchmark.

    This report captures metrics specific to batch concept drift methods,
    focusing on detection accuracy and characteristics of identified drift.
    """

    method: methods.BatchCD  # method used in benchmark
    drift_detected: bool = False  # whether drift was detected
    detection_accuracy: Optional[float] = Field(None, ge=0.0, le=1.0)
    drift_magnitude: Optional[float] = Field(None, ge=0.0)
    confidence_level: Optional[float] = Field(None, ge=0.0, le=1.0)
    drift_location: Optional[Union[int, List[int]]] = None  # est. drift pos.
    testing_power: Optional[float] = Field(None, ge=0.0, le=1.0)  # sta. power

    def __init__(self, criteria: set[Criteria], benchmark: Benchmark) -> None:
        super().__init__(criteria, benchmark)
        if not isinstance(benchmark.method, methods.BatchCD):
            raise ValueError("Method must be a BatchCD instance")
        self.method = benchmark.method
        # TODO: add the rest of the metrics

    @classmethod
    def detect_report_type(cls, data: Dict) -> bool:
        """Detect if the data represents this report type."""
        if not isinstance(data.get("method"), dict):
            return False
        method = data.get("method", {}).get("type", "")
        return "BatchConceptDrift" in str(method)


class BatchDDReport(Report):
    """
    Data class to store the results of a Batch Data Drift Detection
    benchmark.

    This report captures metrics specific to batch data drift detection,
    focusing on statistical measures and feature-level drift analysis.
    """

    method: methods.BatchDD  # method used in benchmark
    drift_detected: bool = False  # whether drift was detected
    drift_score: Optional[float] = Field(None, ge=0.0)  # overall magnitude
    feature_drift_scores: Optional[Dict[str, float]] = None  # per-feature
    p_value: Optional[float] = Field(None, ge=0.0, le=1.0)  # significance
    effect_size: Optional[float] = None  # magnitude of the effect
    confidence_interval: Optional[tuple] = None  # confidence interval

    def __init__(self, criteria: set[Criteria], benchmark: Benchmark) -> None:
        super().__init__(criteria, benchmark)
        if not isinstance(benchmark.method, methods.BatchDD):
            raise ValueError("Method must be a BatchDD instance")
        self.method = benchmark.method
        # TODO: add the rest of the metrics

    @field_validator("confidence_interval")
    @classmethod
    def validate_confidence_interval(cls, v: Any):
        """Validate tuple of length 2 with lower bound ≤ upper bound."""
        if v is not None:
            if not isinstance(v, tuple) or len(v) != 2:
                raise ValueError("Confidence interval must tuple of len 2")
            if v[0] > v[1]:
                raise ValueError("Bound lw must be less or equal to up")
        return v

    @classmethod
    def detect_report_type(cls, data: Dict) -> bool:
        """Detect if the data represents this report type."""
        if not isinstance(data.get("method"), dict):
            return False
        method = data.get("method", {}).get("type", "")
        return "BatchDataDrift" in str(method)
