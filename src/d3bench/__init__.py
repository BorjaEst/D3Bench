"""
This module contains the configuration of the datasets and tools used in the
benchmarking process.
"""

from typing import Type

from d3bench import datasets, tools
from d3bench.config import Criteria, Datafile, Framework
from d3bench.datasets import Dataset
from d3bench.results import Results
from d3bench.tools import Tool

# pylint: disable=too-few-public-methods


# Initialize the datasets constant
DATASETS: dict[Datafile, Dataset] = {
    "energy": datasets.DataEnergy(building_id=1),
    # "occupancy": datasets.DataOccupancy(),
}

# Initialize the tools constant
TOOLS: dict[Framework, Type[Tool]] = {
    "Frouros": tools.Frouros,
    "Evidently": tools.Evidently,
    "NannyML": tools.NannyML,
    "Alibi-Detect": tools.AlibiDetect,
    "River": tools.River,
    "Menelaus": tools.Menelaus,
}
