from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

import nannyml as nml  # pip install nannyml
import numpy as np
import pandas as pd  # pip install pandas
from alibi_detect.cd import CVMDrift, KSDrift, SpotTheDiffDrift
from evidently.metric_preset import DataDriftPreset

# from evidently.metrics import *
from evidently.report import Report
from pydantic import Field

# pylint: disable=too-few-public-methods


class Methods(Enum):
    """Available methods for drift detection."""

    KOLMOGOROV_SMIRNOV = 0  # K-S Test
    WASSERSTEIN = 1  # Wasserstein Distance Normed
    KLD = 2  # Kullback-Leibler divergence
    PSI = 3  # Population Stability Index
    JSD = 4  # Jenson-Shannon Distance
    AD = 5  # Anderson-Darling
    CVM = 6  # Cramer-von-Mises
    HD = 7  # Hellinger distance
    MWURT = 8  # Mann-Whitney U-Rank Test
    ED = 9  # Energy-distance
    ES = 10  # Epps-Singleton
    TT = 11  # T-Test
    SPOTDIFF = 12  # Spot-The-Difference Test


class ToolOptions:
    """Settings to instantiate a tool."""

    show_report: bool = Field(
        default=False,
        description="Show report if set.",
    )


class Tool(ABC):
    """Abstract class for drift detection tools."""

    # Abstract attribute to define by child class
    name: str = ""
    methods: set = set()

    def __init__(self, settings: Optional[ToolOptions] = None):
        settings = settings or ToolOptions()
        self.show_report = settings.show_report

    @abstractmethod
    def __call__(self, df_train, df_test, method):
        """Run drift detection on the reference and current data."""
        raise NotImplementedError

    @abstractmethod
    def preprocess(self, df_reference, df_current):
        """Preprocess the data before drift detection."""
        raise NotImplementedError

    @abstractmethod
    def run_test(self, building_id, test):
        """Run a specific drift detection test."""
        raise NotImplementedError


class Evidently(Tool):
    """Evidently drift detection tool."""

    name = "Evidently"
    methods = {
        Methods.WASSERSTEIN,
        Methods.KLD,
        Methods.PSI,
        Methods.JSD,
        Methods.AD,
        Methods.CVM,
        Methods.HD,
        Methods.MWURT,
        Methods.ED,
        Methods.ES,
        Methods.TT,
        Methods.KOLMOGOROV_SMIRNOV,
    }

    def preprocess(self, args____todo):
        """Preprocess the data before drift detection."""

        # Make a copy of the dataframes
        df_reference, df_current = dataset.splitTrainTest(building_id)
        df_reference, df_current = df_reference.copy(), df_current.copy()

        # If the column names are different, rename them
        if "consumption" in df_reference:
            # Rename the consumption column to target
            df_reference.rename(columns={"consumption": "target"}, inplace=True)
            df_current.rename(columns={"consumption": "target"}, inplace=True)
            # Drop the ids column
            df_reference.drop(columns={"ids"}, inplace=True)
            df_current.drop(columns={"ids"}, inplace=True)
            # Reset the index
            df_reference.reset_index(drop=True, inplace=True)
            df_current.reset_index(drop=True, inplace=True)
            # Convert the columns to numeric
            df_reference["target"] = pd.to_numeric(df_reference["target"])
            df_reference["temp_outside"] = pd.to_numeric(df_reference["temp_outside"])

        # If the columns are probabilities, drop them
        if "prob_predicted" in df_reference:
            # Drop the predicted and probability columns
            df_reference.drop(columns={"prob_predicted", "predicted"}, inplace=True)
            df_current.drop(columns={"prob_predicted", "predicted"}, inplace=True)

        # Return the column names and the dataframes
        return list(df_reference.columns), df_reference, df_current

    def __call__(self, df_reference, df_current, building_id):
        column_names, df_reference, df_current = self.preprocess(
            df_reference, df_current
        )
        my_dict = {}
        for test in self.methods:
            if test == Methods.WASSERSTEIN:
                my_dict["Wasserstein Distanz"] = self.run_test(
                    building_id, "wasserstein"
                )
            elif test == Methods.KLD:
                my_dict["K-L Divergence"] = self.run_test(building_id, "kl_div")
            elif test == Methods.PSI:
                my_dict["PSI"] = self.run_test(building_id, "psi")
            elif test == Methods.JSD:
                my_dict["J-S Distance"] = self.run_test(building_id, "jensenshannon")
            elif test == Methods.AD:
                my_dict["Anderson-Darling"] = self.run_test(building_id, "anderson")
            elif test == Methods.CVM:
                my_dict["Cramer-von-Mises"] = self.run_test(
                    building_id, "cramer_von_mises"
                )
            elif test == Methods.HD:
                my_dict["Hellinger-Distance"] = self.run_test(building_id, "hellinger")
            elif test == Methods.MWURT:
                my_dict["Mann-Whitney U-Rank Test"] = self.run_test(
                    building_id, "mannw"
                )
            elif test == Methods.ED:
                my_dict["Energy-Distance"] = self.run_test(building_id, "ed")
            elif test == Methods.ES:
                try:
                    my_dict["Epps-Singleton"] = self.run_test(building_id, "es")
                except:
                    my_dict["Epps-Singleton"] = "no result"
            elif test == Methods.TT:
                my_dict["T-Test"] = self.run_test(building_id, "t_test")
            elif test == Methods.KOLMOGOROV_SMIRNOV:
                my_dict["K-S Test"] = self.run_test(building_id, "ks")

        return my_dict

    def run_test(self, building_id, test):
        # create TargetDriftReport in dict
        report = Report(metrics=[DataDriftPreset(stattest=test)])
        report.run(reference_data=self.ref, current_data=self.cur)
        if self.showReport:
            file_name = "evidently_report_{}_{}.html".format(building_id, test)
            report.save_html(file_name)
        report_dict = report.as_dict()

        # add into dictionary
        my_dict = {}
        for col in self.column_names:
            my_dict[f"{col}_drift_score"] = report_dict["metrics"][1]["result"][
                "drift_by_columns"
            ][col]["drift_score"]
            my_dict[f"{col}_is_drifted"] = report_dict["metrics"][1]["result"][
                "drift_by_columns"
            ][col]["drift_detected"]

        return my_dict


class NannyML(Tool):
    """NannyML drift detection tool."""

    name = "NannyML"
    methods = {
        Methods.KOLMOGOROV_SMIRNOV,
        Methods.WASSERSTEIN,
        Methods.JSD,
        Methods.HD,
    }

    def preprocess(self):
        if "temp_outside" in self.ref:
            self.ref = self.ref.drop(columns={"ids"})
            self.cur = self.cur.drop(columns={"ids"})
        if "prob_predicted" in self.ref:
            self.ref = self.ref.drop(columns={"prob_predicted", "predicted"})
            self.cur = self.cur.drop(columns={"prob_predicted", "predicted"})
        self.ref["time"] = self.ref.index
        self.ref = self.ref.reset_index(drop=True)
        self.cur["time"] = self.cur.index
        self.cur = self.cur.reset_index(drop=True)
        self.column_names = [col for col in self.ref.columns if col != "time"]

    # @profile
    def __call__(self, ref, cur, building_id):
        self.ref = ref
        self.cur = cur
        self.preprocess()

        my_dict = {}
        for test in self.methods:
            if test == Methods.KOLMOGOROV_SMIRNOV:
                my_dict["K-S Test"] = self.run_test(building_id, "kolmogorov_smirnov")
            elif test == Methods.WASSERSTEIN:
                my_dict["Wasserstein Distance"] = self.run_test(
                    building_id, "wasserstein"
                )
            elif test == Methods.JSD:
                my_dict["J-S Distance"] = self.run_test(building_id, "jensen_shannon")
            elif test == Methods.HD:
                my_dict["Hellinger-Distance"] = self.run_test(building_id, "hellinger")

        return my_dict

    # calculates drift score based on chunks
    # drift score is the mean of all chunks
    # is_drifted is the mean of True/False depending on the threshold, computed to %
    # @profile
    def run_test(self, building_id, test):
        calc = nml.UnivariateDriftCalculator(
            column_names=self.column_names,
            timestamp_column_name="time",
            continuous_methods=[test],
            thresholds={
                "kolmogorov_smirnov": nml.thresholds.StandardDeviationThreshold(
                    std_lower_multiplier=None
                ),
                "jensen_shannon": nml.thresholds.ConstantThreshold(upper=0.1),
                "wasserstein": nml.thresholds.StandardDeviationThreshold(
                    std_lower_multiplier=None
                ),
                "hellinger": nml.thresholds.ConstantThreshold(upper=0.1),
            },
        )

        calc.fit(self.ref)
        results = calc.calculate(self.cur)
        df = results.filter(period="analysis", column_names=self.column_names).to_df()

        if self.showReport:
            figure = results.filter(
                column_names=results.continuous_column_names, methods=[test]
            ).plot(kind="distribution")
            figure.write_image(f"nannyml_report_dist_{building_id}_{test}.svg")
            figure = results.filter(
                column_names=results.continuous_column_names, methods=[test]
            ).plot(kind="drift")
            figure.write_image(f"nannyml_report_drift_{building_id}_{test}.svg")

        # add into dictionary
        my_dict = {}
        for col in self.column_names:
            my_dict[f"{col}_drift_score"] = df[col][test]["value"].mean()
            my_dict[f"{col}_is_drifted"] = (
                str(round(df[col][test]["alert"].mean() * 100, 1)) + " % drifted"
            )

        return my_dict


class AlibiDetect(Tool):

    name = "AlibiDetect"
    methods = {
        Methods.KOLMOGOROV_SMIRNOV,
        Methods.CVM,
        Methods.SPOTDIFF,
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
            if test == Methods.KOLMOGOROV_SMIRNOV:
                my_dict["K-S Test"] = self.run_test("kolmogorov_smirnov")
            elif test == Methods.CVM:
                my_dict["Cramer-von-Mises"] = self.run_test("cramer_von_mises")
            elif test == Methods.SPOTDIFF:
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
