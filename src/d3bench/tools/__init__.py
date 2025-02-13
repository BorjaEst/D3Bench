"""Module to define the drift detection tools."""

from abc import ABC, abstractmethod
from typing import Any, Optional

import pandas as pd
from pydantic import Field

from d3bench.tools import alibi, evidently, frouros, nannyml
from d3bench.utils import Framework, Method

# pylint: disable=too-few-public-methods


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
    methods: dict[Method, Any] = {}

    def __init__(self, settings: Optional[ToolOptions] = None):
        settings = settings or ToolOptions()
        self.show_report = settings.show_report

    @abstractmethod
    def preprocess(self, df: pd.DataFrame) -> Any:
        """Preprocess the data before drift detection."""
        raise NotImplementedError

    def __getitem__(self, name):
        return self.methods[name]


class Frouros(Tool):
    """Frouros drift detection tool."""

    name = "Frouros"
    methods = {
        Method.KOLMOGOROV_SMIRNOV: frouros.KSWIN,
        Method.CVM: frouros.CVMTest,
        # TODO: Add the rest of the methods
    }

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df.drop(columns={"time"}, inplace=True)
        return df


class Evidently(Tool):
    """Evidently drift detection tool."""

    name = "Evidently"
    methods = {
        Method.KOLMOGOROV_SMIRNOV: evidently.KSWIN,
        # Method.WASSERSTEIN: "wasserstein",
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

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df.drop(columns={"time"}, inplace=True)
        return df
