"""Module to define the drift detection tools."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

import pandas as pd
from pydantic import Field
from pydantic_settings import BaseSettings

import d3bench.tools.alibi as tools_alibi
import d3bench.tools.evidently as tools_evidently
import d3bench.tools.frouros as tools_frouros
import d3bench.tools.nannyml as tools_nannyml
from d3bench import benchmarks, methods
from d3bench.config import Data, Framework

# pylint: disable=too-few-public-methods


logger = logging.getLogger(__name__)


class Options(BaseSettings):
    """Settings to instantiate a tool."""

    example_option: str = Field(
        default="example",
        description="Example option for tools.",
    )


class Tool(ABC):
    """Abstract class for drift detection tools."""

    # Abstract attributes to define by child classes
    name: Framework
    online_cd_methods: dict[methods.OnlineCD, Any] = {}
    online_dd_methods: dict[methods.OnlineDD, Any] = {}
    batch_cd_methods: dict[methods.BatchCD, Any] = {}
    batch_dd_methods: dict[methods.BatchDD, Any] = {}

    def __init__(self, data: Data, settings: Optional[Options] = None):
        settings = settings or Options()
        self.x_reference = self.preprocess(data["x_reference"].copy())
        self.x_test = self.preprocess(data["x_test"].copy())

    @abstractmethod
    def preprocess(self, df: pd.DataFrame) -> Any:
        """Preprocess the data before drift detection."""
        raise NotImplementedError

    def process(self, df: pd.DataFrame) -> Any:
        """Call to the preprocess method with a copy of the data."""
        return self.preprocess(df.copy())

    @property
    def data(self) -> Data:
        """Return the data used for drift detection."""
        return {
            "x_reference": self.x_reference,
            "x_test": self.x_test,
        }


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


class AlibiDetect(Tool):
    """Alibi-Detect drift detection tool."""

    name: Framework = "Alibi-Detect"
    online_cd_methods: dict[methods.OnlineCD, Any] = {}
    online_dd_methods: dict[methods.OnlineDD, Any] = {}
    batch_dd_methods: dict[methods.BatchDD, Any] = {}

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError
