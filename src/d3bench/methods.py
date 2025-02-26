"""
Drift detection method definitions and categorization for the package.

This module provides a comprehensive taxonomy of drift detection methods,
organized by their characteristics, requirements, and use cases.
"""

from enum import StrEnum


# Online Supervised Concept Drift Detection
class OnlineCD(StrEnum):
    """
    Definition: These methods monitor prediction errors in a supervised
        learning model to detect when the relationship between input and
        output changes.
    Comparison Factors: Drift detection speed, false positive rate, memory
        efficiency.
    Use Cases: Model monitoring, fraud detection, real-time anomaly
        detection.
    """

    # Change Detection
    BOCD = "Bayesian Online Change Detection"
    CUSUM = "Cumulative Sum Control Chart"
    GMA = "Geometric Moving Average"
    PCA_CD = "Principal Component Analysis for Concept Drift"
    PHT = "Page-Hinkley Test"

    # Statistical Process Control
    DDM = "Drift Detection Method"
    ECDD = "Exponential Cumulative Drift Detection"
    ECDD_H = "Ensemble-based Exponential Cumulative Drift Detection"
    ECDDWT = "EWMA Concept Drift Detection Warning"
    EDDM = "Early Drift Detection Method"
    HDDM_A = "Hoeffding's Drift Detection Method Test-A"
    HDDM_W = "Hoeffding's Drift Detection Method Test-W"
    RDDM = "Reactive Drift Detection Method"

    # Window Based
    ADWIN = "Adaptive Windowing"
    KSWIN = "Kolmogorov-Smirnov Windowing"
    OAUE = "Online Accuracy Updated Ensemble"
    STEPD = "Statistical Test of Equal Proportions Detection"

    # Other
    CDDM = "Change Detection in Data Streams"


# Online Unsupervised Data Drift Detection
class OnlineDD(StrEnum):
    """
    Definition: These methods detect distribution changes in input features
        without needing labels.
    Comparison Factors: Sensitivity to small distribution changes, robustness
        against noise.
    Use Cases: Feature monitoring, drift in unlabelled data, preemptive
        retraining triggers.
    """

    # Distance Based
    DDMO = "Drift Detection Model Output"
    MMD = "Maximum Mean Discrepancy"
    UDD = "Unsupervised Drift Detector"

    # Statistical Test
    LFR = "Log-Likelihood Ratio Test"
    KSI = "Incremental Kolmogorov-Smirnov Test"


# Batch Concept Drift Detection
class BatchCD(StrEnum):
    """
    Definition: These methods analyze model predictions periodically (not in
        real-time) to detect changes in the feature-label relationship.
    Comparison Factors: Detection lag, computational efficiency, precision.
    Use Cases: Periodic model validation, offline analysis of model decay.
    """


# Batch Data Drift Detection
class BatchDD(StrEnum):
    """
    Definition: These methods detect changes in feature distributions in large
        batches of data.
    Comparison Factors: Performance on high-dimensional data, ability to
        detect local vs. global drift.
    Use Cases: Data validation, detecting changes in training data before
        model retraining.
    """

    # Distance Based
    BHATTACHARYYA = "Bhattacharyya Distance"
    EMD = "Earth Mover's Distance"
    ENERGY = "Energy Distance"
    HELLINGER = "Hellinger Distance"
    HI_NCOMP = "Histogram Intersection Normalized Complement"
    JSD = "Jensen-Shannon Divergence Drift Detection"
    KLD = "Kullback-Leibler Divergence Drift Detection"
    LSDD = "Least-Squares Density Difference"
    MMD = "Maximum Mean Discrepancy"
    PSI = "Population Stability Index"

    # Statistical Test
    ANDERSON_DARLING = "Anderson-Darling Test"
    BWS = "Baumgartner Weiss Schindler Test"
    CHI_SQUARE = "Chi-square Test"
    CVM = "Cramér-von Mises Test"
    KS = "Kolmogorov-Smirnov Test"
    KUIPER = "Kuiper's Test"
    MANN_WHITNEY = "Mann-Whitney U-Test"
    MDDM_A = "Mc Diarmid Drift Detection Method Test-A"
    MDDM_E = "Mc Diarmid Drift Detection Method Test-E"
    MDDM_G = "Mc Diarmid Drift Detection Method Test-G"
    WELCH_T = "Welch's T-Test"
