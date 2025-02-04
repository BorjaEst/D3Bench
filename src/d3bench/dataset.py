import numpy as np
import pandas as pd

from d3bench import config


class Dataset:

    def __init__(self, path):
        self.df = pd.read_csv(path)

    def preprocess(self):
        pass

    def splitTrainTest(self, building_id):
        pass


class Data_Energy(Dataset):

    def __init__(self, path):
        super().__init__(path)
        self.preprocess()

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
        self.df = self.df[self.df["time"] >= config.DATA_START_DATE]
        self.df = self.df[self.df["time"] < config.DATA_END_DATE]

    def splitTrainTest(self, building_id):
        df_group = self.createGroup(building_id)
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

        return train_set, test_set

    def returnBuilding(self, building_id):
        df_group = self.createGroup(building_id)
        df_group = df_group.set_index("time")
        df_group.index = pd.to_datetime(df_group.index)
        df_group = df_group.groupby(pd.Grouper(freq="h")).sum()

        return df_group

    def createGroup(self, building_id):
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


class Data_Occupancy(Dataset):

    def __init__(self, path):
        super().__init__(path)
        self.preprocess()

    def preprocess(self):
        df = self.df
        if "time" in df:
            # convert to datetime
            df["time"] = pd.to_datetime(df["time"])

        # self.df = self.df.drop(columns=['Unnamed: 0'], axis=1)

    def splitTrainTest(self, building_id):
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
