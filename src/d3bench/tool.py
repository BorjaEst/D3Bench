from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any, Literal, Optional

import evidently.metric_preset
import evidently.report
import nannyml as nml
import numpy.typing as npt
import pandas as pd
from alibi_detect import cd
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
    def preprocess(self, df: pd.DataFrame, store: dict[str, Any]) -> Any:
        """Preprocess the data before drift detection."""
        raise NotImplementedError

    @abstractmethod
    def setup(self, method: Method, store: dict[str, Any]) -> None:
        """Setup the drift detection method."""
        raise NotImplementedError

    @abstractmethod
    def fit(self, x_reference: Any, store: dict[str, Any]) -> None:
        """Run drift detection on the reference and current data."""
        raise NotImplementedError

    @abstractmethod
    def test(self, x_test: Any, store: dict[str, Any]) -> None:
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
        Method.CVM: "cramer_von_mises",
        # Method.HD: "hellinger",
        # Method.MWURT: "mannw",
        # Method.ED: "ed",
        # Method.ES: "es",
        # Method.TT: "t_test",
    }

    def preprocess(self, df: pd.DataFrame, store: dict[str, Any]) -> Any:
        if "consumption" in df:
            df.rename(columns={"consumption": "target"}, inplace=True)
            df.drop(columns={"ids"}, inplace=True)
            df.reset_index(drop=True, inplace=True)
            df["target"] = pd.to_numeric(df["target"])
            df["temp_outside"] = pd.to_numeric(df["temp_outside"])
        if "prob_predicted" in df:
            df.drop(columns={"prob_predicted", "predicted"}, inplace=True)
        return df

    def setup(self, method: Method, store: dict[str, Any]) -> None:
        test = self.methods[method]
        metrics = [evidently.metric_preset.DataDriftPreset(stattest=test)]
        store["report"] = evidently.report.Report(metrics)

    def fit(self, x_reference: pd.DataFrame, store: dict[str, Any]) -> None:
        # There is no train method separated from the test method in Evidetly
        store["df_reference"] = x_reference
        raise NotImplementedError("Evidently does not have a train method")

    def test(self, x_test: pd.DataFrame, store: dict[str, Any]) -> None:
        report, x_reference = store["report"], store["df_reference"]
        report.run(reference_data=x_reference, current_data=x_test)

    def postprocess(self, store: dict[str, Any]) -> dict[str, Any]:
        return store["report"].as_dict()


class NannyML(Tool):
    """NannyML drift detection tool."""

    name = "NannyML"
    methods = {
        Method.KOLMOGOROV_SMIRNOV: "kolmogorov_smirnov",
        Method.WASSERSTEIN: "wasserstein",
        # Method.JSD: "jensen_shannon",
        # Method.HD: "hellinger",
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

    def preprocess(self, df: pd.DataFrame, store: dict[str, Any]) -> Any:
        if "temp_outside" in df:
            df.drop(columns={"ids"}, inplace=True)
        if "prob_predicted" in df:
            df.drop(columns={"prob_predicted", "predicted"}, inplace=True)
        df["time"] = df.index
        df.reset_index(drop=True, inplace=True)
        if "column_names" not in store:
            store["column_names"] = df.columns
        return df

    def setup(self, method: Method, store: dict[str, Any]) -> None:
        store["detector"] = nml.UnivariateDriftCalculator(
            column_names=store["column_names"],
            timestamp_column_name="time",
            continuous_methods=[self.methods[method]],
            thresholds=self.thresholds,
        )

    def fit(self, x_reference: pd.DataFrame, store: dict[str, Any]) -> None:
        store["detector"].fit(x_reference)

    def test(self, x_test: pd.DataFrame, store: dict[str, Any]) -> None:
        store["results"] = store["detector"].calculate(x_test)

    def postprocess(self, store: dict[str, Any]) -> dict[str, Any]:
        return (
            store["results"]
            .filter(period="analysis", column_names=store["column_names"])
            .to_dict()
        )


class AlibiDetect(Tool):
    """Alibi Detect drift detection tool."""

    name = "Alibi-Detect"
    methods = {
        Method.KOLMOGOROV_SMIRNOV: cd.KSDrift,
        Method.CVM: cd.CVMDrift,
        # Method.SPOTDIFF: cd.SpotTheDiffDrift,
    }

    def preprocess(self, df: pd.DataFrame, store: dict[str, Any]) -> Any:
        if "prob_predicted" in df:
            df.drop(columns={"prob_predicted", "predicted"}, inplace=True)
        if "consumption" in df:
            df.drop(columns={"ids"}, inplace=True)
        store["column_names"] = df.columns
        return df.to_numpy()

    def setup(self, method: Method, store: dict[str, Any]) -> None:
        store["d_class"] = self.methods[method]

    def fit(self, x_reference: npt.NDArray[Any], store: dict[str, Any]) -> None:
        store["detector"] = store["d_class"](x_ref=x_reference)

    def test(self, x_test: npt.NDArray[Any], store: dict[str, Any]) -> None:
        options = {"drift_type": "feature", "return_p_val": True}
        store["results"] = store["detector"].predict(x_test, **options)

    def postprocess(self, store: dict[str, Any]) -> dict[str, Any]:
        return store["results"]["data"].to_dict()
