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


Dataset = Literal["energy", "occupancy"]
Data = Tuple[pd.DataFrame, pd.DataFrame]

DATA_PATH = Path("data")


class Datasets(Enum):
    """Enum class for benchmark datasets"""

    ENERGY = "energy"
    OCCUPANCY = "occupancy"


class DatasetOptions(BaseSettings):
    """Settings to instantiate a dataset."""

    data_start: dt.date = Field(
        default=dt.date(2019, 4, 1),
        description="Start date.",
    )
    data_end: dt.date = Field(
        default=dt.date(2022, 4, 1),
        description="End date.",
    )


class BaseDataset(ABC):

    file_name: str

    def __init__(self, settings: Optional[DatasetOptions] = None):
        settings = settings or DatasetOptions()
        self.df = pd.read_csv(DATA_PATH / self.file_name)
        self.data_start = settings.data_start
        self.data_end = settings.data_end
        self.preprocess()

    @abstractmethod
    def preprocess(self):
        """Preprocess the dataset."""
        raise NotImplementedError

    def __call__(self, building_id):
        """Return the dataset for the given building."""
        return self.split_data(building_id)

    @abstractmethod
    def split_data(self, building_id):
        """Split the dataset into reference and current sets."""
        raise NotImplementedError


class DataEnergy(BaseDataset):

    file_name = "energy_data.csv"

    def preprocess(self, datetime_columns=None):

        # Merge date and time columns to datetime column
        datetime_columns = datetime_columns or ["year", "month", "day", "hour"]
        self.df["time"] = pd.to_datetime(self.df[datetime_columns])
        self.df = self.df.drop(columns=datetime_columns)

        # Drop duplicates, sort and reset index
        self.df.drop_duplicates(subset=["ids", "time"], inplace=True)
        self.df.sort_values(by=["ids", "time"], inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        # Drop non used columns and negative consumption values
        self.df = self.df[["ids", "time", "consumption", "temp_outside"]]
        self.df = self.df[self.df["consumption"] >= 0]

        # Use only data defined by the config
        self.df = self.df[self.df["time"] >= str(self.data_start)]
        self.df = self.df[self.df["time"] <= str(self.data_end)]

    def split_data(self, building_id):
        df_group = self._create_group(building_id)
        boundary = "04-01-2020 00:00"

        # Split dataset into train and test
        train_set = df_group.loc[(df_group["time"] < boundary)]
        train_set = train_set.set_index("time")
        train_set.index = pd.to_datetime(train_set.index)
        train_set = train_set.groupby(pd.Grouper(freq="h")).sum()

        test_set = df_group.loc[df_group["time"] >= boundary]
        test_set = test_set.set_index("time")
        test_set.index = pd.to_datetime(test_set.index)
        test_set = test_set.groupby(pd.Grouper(freq="h")).sum()

        # Return train and test sets
        return train_set, test_set

    def _create_group(self, building_id):
        df = self.df
        df_group = df[df["ids"] == building_id]
        num_wrongs = np.sum(df_group["consumption"] < 0)
        if num_wrongs:
            df_group.loc[df_group["consumption"] < 0, "consumption"] = (
                np.nan
            )  # if there are values <0, they are stated as nan
            df_group["consumption"] = df_group[
                "consumption"
            ].interpolate()  # and linear interpolated
        return df_group


class DataOccupancy(BaseDataset):

    file_name = "occupancy_data.csv"

    def preprocess(self):
        df = self.df
        if "time" in df:
            # convert to datetime
            df["time"] = pd.to_datetime(df["time"])

        # self.df = self.df.drop(columns=['Unnamed: 0'], axis=1)

    def split_data(self, building_id):
        df_group = self.df
        df_group["time"] = pd.to_datetime(df_group["time"])
        boundary = pd.to_datetime("05-09-2021 00:00")

        train_set = df_group.loc[(df_group["time"] < boundary)]
        train_set = train_set.set_index("time")
        train_set.index = pd.to_datetime(train_set.index)

        test_set = df_group.loc[df_group["time"] >= boundary]
        test_set = test_set.set_index("time")
        test_set.index = pd.to_datetime(test_set.index)

        return train_set, test_set
