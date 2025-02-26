"""Module to define the drift detection tools."""

from abc import ABC, abstractmethod
from typing import Any, Optional

import pandas as pd
from pydantic import Field

import d3bench.tools.alibi as tools_alibi
import d3bench.tools.evidently as tools_evidently
import d3bench.tools.frouros as tools_frouros
import d3bench.tools.nannyml as tools_nannyml
from d3bench import methods
from d3bench.config import Framework

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
    online_cd_methods: dict[methods.OnlineCD, Any] = {}
    online_dd_methods: dict[methods.OnlineDD, Any] = {}
    batch_dd_methods: dict[methods.BatchDD, Any] = {}

    def __init__(self, settings: Optional[ToolOptions] = None):
        settings = settings or ToolOptions()
        self.show_report = settings.show_report

    @abstractmethod
    def preprocess(self, df: pd.DataFrame) -> Any:
        """Preprocess the data before drift detection."""
        raise NotImplementedError


class Frouros(Tool):
    """Frouros drift detection tool."""

    name: Framework = "Frouros"
    online_cd_methods: dict[methods.OnlineCD, Any] = {}
    online_dd_methods: dict[methods.OnlineDD, Any] = {}
    batch_dd_methods: dict[methods.BatchDD, Any] = {}

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df.drop(columns={"time"}, inplace=True)
        return df


class Evidently(Tool):
    """Evidently drift detection tool."""

    name: Framework = "Evidently"
    online_cd_methods: dict[methods.OnlineCD, Any] = {}
    online_dd_methods: dict[methods.OnlineDD, Any] = {}
    batch_dd_methods: dict[methods.BatchDD, Any] = {}

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df.drop(columns={"time"}, inplace=True)
        return df


class NannyML(Tool):
    """NannyML drift detection tool."""

    name: Framework = "NannyML"
    online_cd_methods: dict[methods.OnlineCD, Any] = {}
    online_dd_methods: dict[methods.OnlineDD, Any] = {}
    batch_dd_methods: dict[methods.BatchDD, Any] = {}

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return df  # No preprocessing needed
