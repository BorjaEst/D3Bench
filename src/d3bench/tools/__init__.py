"""Module to define the drift detection tools."""

import logging
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, Generator, Optional

import pandas as pd
from pydantic import Field
from pydantic_settings import BaseSettings

import d3bench.tools.alibi as tools_alibi
import d3bench.tools.evidently as tools_evidently
import d3bench.tools.frouros as tools_frouros
import d3bench.tools.nannyml as tools_nannyml
from d3bench import  methods
from d3bench.config import Criteria, Framework
from d3bench.reports import Report
from d3bench.utils import Data

# pylint: disable=too-few-public-methods


logger = logging.getLogger(__name__)


class Options(BaseSettings):
    """Settings to instantiate a tool."""

    repetitions: int = Field(
        default=3,
        description="Number of repetitions for the benchmark.",
    )

    on_vm: bool = Field(
        default=False,
        description="Flag to run the benchmark on a VM.",
    )


class Tool(ABC):
    """Abstract class for drift detection tools."""

    # Abstract attributes to define by child classes
    name: Framework
    online_cd_methods: dict[methods.OnlineCD, Any]
    online_dd_methods: dict[methods.OnlineDD, Any]
    batch_cd_methods: dict[methods.BatchCD, Any]
    batch_dd_methods: dict[methods.BatchDD, Any]

    def __init__(self, data: Data, settings: Optional[Options] = None):
        self.settings = settings or Options()
        self.data = data

    @abstractmethod
    def preprocess(self, df: pd.DataFrame) -> Any:
        """Preprocess the data before drift detection."""
        raise NotImplementedError

    def process(self, df: pd.DataFrame) -> Any:
        """Call to the preprocess method with a copy of the data."""
        return self.preprocess(df.copy())

    @cached_property
    def reference_data(self) -> Any:
        """Return the reference data."""
        return self.preprocess(self.data.reference.copy())

    @cached_property
    def testing_data(self) -> Any:
        """Return the testing data."""
        return self.preprocess(self.data.testing.copy())


class Frouros(Tool):
    """Frouros drift detection tool."""

    name: Framework = "Frouros"
    online_cd_methods: dict[methods.OnlineCD, Any] = {}
    online_dd_methods: dict[methods.OnlineDD, Any] = {}
    batch_cd_methods: dict[methods.BatchCD, Any] = {}
    batch_dd_methods: dict[methods.BatchDD, Any] = {
        methods.BatchDD.KS: tools_frouros.KSTest,
    }

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df.drop(columns={"time"}, inplace=True)
        return [df[feature].to_numpy() for feature in df.columns]


class Evidently(Tool):
    """Evidently drift detection tool."""

    name: Framework = "Evidently"
    online_cd_methods: dict[methods.OnlineCD, Any] = {}
    online_dd_methods: dict[methods.OnlineDD, Any] = {}
    batch_cd_methods: dict[methods.BatchCD, Any] = {}
    batch_dd_methods: dict[methods.BatchDD, Any] = {
        methods.BatchDD.KS: tools_evidently.KSTest,
    }

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df.drop(columns={"time"}, inplace=True)
        return df


class NannyML(Tool):
    """NannyML drift detection tool."""

    name: Framework = "NannyML"
    online_cd_methods: dict[methods.OnlineCD, Any] = {}
    online_dd_methods: dict[methods.OnlineDD, Any] = {}
    batch_cd_methods: dict[methods.BatchCD, Any] = {}
    batch_dd_methods: dict[methods.BatchDD, Any] = {
        methods.BatchDD.KS: tools_nannyml.KSTest,
    }

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return df  # No preprocessing needed


class AlibiDetect(Tool):
    """Alibi-Detect drift detection tool."""

    name: Framework = "Alibi-Detect"
    online_cd_methods: dict[methods.OnlineCD, Any] = {}
    online_dd_methods: dict[methods.OnlineDD, Any] = {}
    batch_cd_methods: dict[methods.BatchCD, Any] = {}
    batch_dd_methods: dict[methods.BatchDD, Any] = {
        # methods.BatchDD.KS: tools_alibi.KSTest,
    }

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError
