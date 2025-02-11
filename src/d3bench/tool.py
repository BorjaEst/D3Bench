from abc import ABC, abstractmethod
from enum import Enum, StrEnum
from typing import Any, Literal, Optional

import nannyml as nml
import numpy as np
import pandas as pd
from alibi_detect.cd import CVMDrift, KSDrift, SpotTheDiffDrift
from evidently.metric_preset import DataDriftPreset

# from evidently.metrics import *
from evidently.report import Report as EvidentlyReport
from pydantic import Field

# pylint: disable=too-few-public-methods
Framework = Literal["Evidently", "NannyML", "Alibi-Detect"]


class Method(StrEnum):
    """Available methods for drift detection."""

    KOLMOGOROV_SMIRNOV = "K-S Test"
    WASSERSTEIN = "Wasserstein Distanz"
    KLD = "K-L Divergence"
    PSI = "PSI"
    JSD = "J-S Distance"
    AD = "Anderson-Darling"
    CVM = "Cramer-von-Mises"
    HD = "Hellinger-Distance"
    MWURT = "Mann-Whitney U-Rank Test"
    ED = "Energy-Distance"
    ES = "Epps-Singleton"
    TT = "T-Test"
    SPOTDIFF = "Spot-The-Difference Test"


class ToolOptions:
    """Settings to instantiate a tool."""

    show_report: bool = Field(
        default=False,
        description="Show report if set.",
    )


class Tool(ABC):
    """Abstract class for drift detection tools."""

    # Abstract attribute to define by child class
    name: Framework
    methods: dict[Method, str] = {}

    def __init__(self, settings: Optional[ToolOptions] = None):
        settings = settings or ToolOptions()
        self.show_report = settings.show_report

    @abstractmethod
    def preprocess(self, df: pd.DataFrame, store: dict[str, Any]) -> None:
        """Preprocess the data before drift detection."""
        raise NotImplementedError

    @abstractmethod
    def setup(self, method: Method, store: dict[str, Any]) -> None:
        """Setup the drift detection method."""
        raise NotImplementedError

    @abstractmethod
    def fit(self, df_reference: pd.DataFrame, store: dict[str, Any]) -> None:
        """Run drift detection on the reference and current data."""
        raise NotImplementedError

    @abstractmethod
    def test(self, df_test: pd.DataFrame, store: dict[str, Any]) -> None:
        """Run drift detection on the reference and current data."""
        raise NotImplementedError

    @abstractmethod
    def postprocess(self, store: dict[str, Any]) -> dict[str, Any]:
        """Postprocess the drift detection results."""
        raise NotImplementedError


class Evidently(Tool):
    """Evidently drift detection tool."""

    name = "Evidently"
    methods = {
        Method.KOLMOGOROV_SMIRNOV: "ks",
        Method.WASSERSTEIN: "wasserstein",
        # Method.KLD: "kl_div",
        # Method.PSI: "psi",
        # Method.JSD: "jensenshannon",
        # Method.AD: "anderson",
        # Method.CVM: "cramer_von_mises",
        # Method.HD: "hellinger",
        # Method.MWURT: "mannw",
        # Method.ED: "ed",
        # Method.ES: "es",
        # Method.TT: "t_test",
    }

    def preprocess(self, df: pd.DataFrame, store: dict[str, Any]) -> None:
        if "consumption" in df:
            df.rename(columns={"consumption": "target"}, inplace=True)
            df.drop(columns={"ids"}, inplace=True)
            df.reset_index(drop=True, inplace=True)
            df["target"] = pd.to_numeric(df["target"])
            df["temp_outside"] = pd.to_numeric(df["temp_outside"])
        if "prob_predicted" in df:
            df.drop(columns={"prob_predicted", "predicted"}, inplace=True)

    def setup(self, method: Method, store: dict[str, Any]) -> None:
        metrics = [DataDriftPreset(stattest=self.methods[method])]
        store["report"] = EvidentlyReport(metrics)

    def fit(self, df_reference: pd.DataFrame, store: dict[str, Any]) -> None:
        # There is no train method separated from the test method in Evidetly
        store["df_reference"] = df_reference
        raise NotImplementedError("Evidently does not have a train method")

    def test(self, df_test: pd.DataFrame, store: dict[str, Any]) -> None:
        report, df_reference = store["report"], store["df_reference"]
        report.run(reference_data=df_reference, current_data=df_test)

    def postprocess(self, store: dict[str, Any]) -> dict[str, Any]:
        return store["report"].as_dict()


class NannyML(Tool):
    """NannyML drift detection tool."""

    name = "NannyML"
    methods = {
        Method.KOLMOGOROV_SMIRNOV: "kolmogorov_smirnov",
        Method.WASSERSTEIN: "wasserstein",
        Method.JSD: "jensen_shannon",
        Method.HD: "hellinger",
    }
    thresholds = {  # TODO: Check if individual thresholds improves runtime
        "kolmogorov_smirnov": nml.thresholds.StandardDeviationThreshold(
            std_lower_multiplier=None,
        ),
        "jensen_shannon": nml.thresholds.ConstantThreshold(
            upper=0.1,
        ),
        "wasserstein": nml.thresholds.StandardDeviationThreshold(
            std_lower_multiplier=None,
        ),
        "hellinger": nml.thresholds.ConstantThreshold(
            upper=0.1,
        ),
    }

    def preprocess(self, df: pd.DataFrame, store: dict[str, Any]) -> None:
        if "temp_outside" in df:
            df.drop(columns={"ids"}, inplace=True)
        if "prob_predicted" in df:
            df.drop(columns={"prob_predicted", "predicted"}, inplace=True)
        df["time"] = df.index
        df.reset_index(drop=True, inplace=True)
        if "column_names" not in store:
            store["column_names"] = list(df)

    def setup(self, method: Method, store: dict[str, Any]) -> None:
        store["detector"] = nml.UnivariateDriftCalculator(
            column_names=store["column_names"],
            timestamp_column_name="time",
            continuous_methods=[self.methods[method]],
            thresholds=self.thresholds,
        )

    def fit(self, df_reference: pd.DataFrame, store: dict[str, Any]) -> None:
        store["detector"].fit(df_reference)

    def test(self, df_test: pd.DataFrame, store: dict[str, Any]) -> None:
        store["results"] = store["detector"].calculate(df_test)

    def postprocess(self, store: dict[str, Any]) -> dict[str, Any]:
        return (
            store["results"]
            .filter(period="analysis", column_names=store["column_names"])
            .to_dict()
        )


class AlibiDetect(Tool):

    name = "AlibiDetect"
    methods = {
        Method.KOLMOGOROV_SMIRNOV,
        Method.CVM,
        Method.SPOTDIFF,
    }

    def preprocess(self):
        if "prob_predicted" in self.ref:
            self.ref = self.ref.drop(columns={"predicted", "prob_predicted"})
            self.cur = self.cur.drop(columns={"predicted", "prob_predicted"})
        if "consumption" in self.ref:
            self.ref = self.ref.drop(columns={"ids"})
            self.cur = self.cur.drop(columns={"ids"})
        self.column_names = list(self.cur)
        self.ref = self.ref.to_numpy()
        self.cur = self.cur.to_numpy()

    def __call__(self, ref, cur, building_id):
        self.ref = ref
        self.cur = cur
        self.preprocess()

        my_dict = {}
        for test in self.methods:
            if test == Method.KOLMOGOROV_SMIRNOV:
                my_dict["K-S Test"] = self.run_test("kolmogorov_smirnov")
            elif test == Method.CVM:
                my_dict["Cramer-von-Mises"] = self.run_test("cramer_von_mises")
            elif test == Method.SPOTDIFF:
                my_dict["Spot-the-diff"] = self.run_test("spotdiff")

        return my_dict

    def run_test(self, test):
        report_dict = {}
        my_dict = {}

        if test == "kolmogorov_smirnov":
            cd = KSDrift(x_ref=self.ref)
            report_dict = cd.predict(self.cur, drift_type="feature", return_p_val=True)
            for i in range(len(self.column_names)):
                col = self.column_names[i]
                my_dict[f"{col}_drift_score"] = report_dict["data"]["p_val"][i]
                my_dict[f"{col}_is_drifted"] = report_dict["data"]["is_drift"][i]
        elif test == "cramer_von_mises":
            cd = CVMDrift(x_ref=self.ref)
            report_dict = cd.predict(self.cur, drift_type="feature", return_p_val=True)
            for i in range(len(self.column_names)):
                col = self.column_names[i]
                my_dict[f"{col}_drift_score"] = report_dict["data"]["p_val"][i]
                my_dict[f"{col}_is_drifted"] = report_dict["data"]["is_drift"][i]
        elif test == "spotdiff":
            self.ref, self.cur = np.asarray(self.ref, np.float32), np.asarray(
                self.cur, np.float32
            )
            cd = SpotTheDiffDrift(x_ref=self.ref)
            report_dict = cd.predict(self.cur, return_p_val=True)
            score = report_dict["data"]["p_val"]
            drifted = report_dict["data"]["is_drift"]
            my_dict = {"drift_score": score, "is_drifted": drifted}

        return my_dict
