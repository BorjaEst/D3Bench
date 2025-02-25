"""
This module contains the configuration of the datasets and tools used in the 
benchmarking process.
"""

import d3bench.dataset as ds
import d3bench.tools as tl
from d3bench.config import Dataset, Framework

DATASETS: dict[Dataset, ds.Dataset] = {
    "energy": ds.DataEnergy(building_id=1),
    # "occupancy": ds.DataOccupancy(),
}

TOOLS: dict[Framework, tl.Tool] = {
    "Frouros": tl.Frouros(),
    "Evidently": tl.Evidently(),
    "NannyML": tl.NannyML(),
    # "Alibi-Detect": tl.AlibiDetect(),
}
