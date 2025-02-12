import datetime as dt
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple, Literal

import numpy as np
import pandas as pd
from pydantic import Field
from pydantic_settings import BaseSettings

# pylint: disable=too-few-public-methods


Data = Tuple[pd.DataFrame, pd.DataFrame]
DATA_PATH = Path("data")


class Datasets(Enum):
    """Enum class for benchmark datasets"""

    ENERGY = "energy"
    OCCUPANCY = "occupancy"


class DatasetOptions(BaseSettings):
    """Settings to instantiate a dataset."""

    buildings: Optional[set[int]] = Field(
        default=None,
        description="List of buildings to use.",
    )
    data_start: dt.date = Field(
        default=dt.date(2019, 4, 1),
        description="Start date.",
    )
    data_end: dt.date = Field(
        default=dt.date(2022, 4, 1),
        description="End date.",
    )
    boundary: dt.date = Field(
        default=dt.date(2020, 4, 1),
        description="Boundary date.",
    )


class Dataset(ABC):
    """Abstract class for the datasets."""

    file_name: str

    def __init__(self, settings: Optional[DatasetOptions] = None):
        settings = settings or DatasetOptions()
        self.df = pd.read_csv(DATA_PATH / self.file_name)
        self.buildings = settings.buildings
        self.data_start = settings.data_start
        self.data_end = settings.data_end
        self.boundary = settings.boundary
        self.preprocess()
        self.x_reference, self.x_test = self.split_data()

    @abstractmethod
    def preprocess(self):
        """Preprocess the dataset."""
        raise NotImplementedError

    def __call__(self):
        """Return the dataset for the given building."""
        return self.split_data()

    @abstractmethod
    def split_data(self):
        """Split the dataset into reference and current sets."""
        raise NotImplementedError


class DataEnergy(Dataset):
    """Class for the energy dataset."""

    file_name = "energy_data.csv"

    def preprocess(self):
        # Merge date and time columns to datetime column
        datetime_columns = ["year", "month", "day", "hour"]
        self.df["time"] = pd.to_datetime(self.df[datetime_columns])
        self.df = self.df.drop(columns=datetime_columns)

        # Drop duplicates, sort and reset index
        self.df.drop_duplicates(subset=["ids", "time"], inplace=True)
        self.df.sort_values(by=["ids", "time"], inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        # Drop rows with building ids not in the list
        self.df = self.df[self.df["ids"].isin(self.buildings)]

        # Drop non used columns and negative consumption values
        self.df = self.df[["ids", "time", "consumption"]]
        self.df = self.df[self.df["consumption"] >= 0]
        self.df.rename(columns={"consumption": "target"}, inplace=True)

        # Use only data defined by the config
        self.df = self.df[self.df["time"] >= str(self.data_start)]
        self.df = self.df[self.df["time"] <= str(self.data_end)]

    def split_data(self):
        # Split data based on the boundary date
        boundary_timestamp = pd.Timestamp(self.boundary)
        train_set = self.df[self.df["time"] < boundary_timestamp]
        test_set = self.df[self.df["time"] >= boundary_timestamp]

        # Return train and test sets
        return train_set, test_set
