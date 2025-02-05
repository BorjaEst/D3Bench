"""
This module provides a command-line interface to run D3Bench benchmarks.

The script allows users to specify various parameters for the benchmark,
including the buildings to benchmark, the criteria to test, the tools to use,
whether to run on a VM, the dataset to use, and the logging level.
"""

import argparse
import logging
import datetime as dt

from d3bench import config
from d3bench.benchmark import Benchmark, Criteria
from d3bench.dataset import Data_Energy, Data_Occupancy
from d3bench.tool import AlibiDetect, Evidently, NannyML


logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="Available tests: FUNCTIONAL, RUNTIME, CPU_RUNTIME, STORAGE\n"
    "Available tools: Evidently, NannyML, AlibiDetect",
)
parser.add_argument(
    "--log-level",
    type=str,
    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    default="INFO",
    help="Logging level (default: INFO).",
)
parser.add_argument(
    "--show-report",
    action="store_true",
    help="Show report if set, otherwise do not show report.",
)
parser.add_argument(
    "--buildings",
    type=int,
    nargs="+",
    default=[1],
    help="List of building IDs to benchmark.",
)
parser.add_argument(
    "--tests",
    type=str,
    nargs="+",
    default=["FUNCTIONAL", "RUNTIME", "CPU_RUNTIME", "STORAGE"],
    help="List of criteria to test.",
)
parser.add_argument(
    "--tools",
    type=str,
    nargs="+",
    default=["Evidently", "NannyML", "AlibiDetect"],
    help="List of tools to benchmark.",
)
parser.add_argument(
    "--vm",
    action="store_true",
    help="Run on VM if set, otherwise run locally.",
)
parser.add_argument(
    "--dataset",
    type=str,
    choices=["energy", "occupancy"],
    default="energy",
    help="Dataset to use (default: energy).",
)
parser.add_argument(
    "--report-id",
    type=str,
    help="Report ID for the benchmark.",
)


def main(args):
    """Run the benchmark with the given arguments."""

    # Print benchmark start message
    print("---------Benchmark execution started---------")

    # Set the logging level from the arguments
    logger.setLevel(args.log_level)
    logger.debug("args: %s", args)

    # Define the criteria and buildings to test
    datetime = dt.datetime.now().strftime("%Y%m%d%H%M%S")
    report_id = args.report_id or f"report_{datetime}"
    tests = [Criteria[criterion] for criterion in args.tests]
    buildings = set(args.buildings)

    # Run the benchmark with the given parameters
    for tool in set(args.tools):
        logger.info("Running benchmark for tool: %s", tool)
        run_benchmark(report_id, tool, tests, buildings, args)

    # Print the benchmark end message
    print("---------Benchmark execution completed---------")


# one benchmark execution with given criteria, tools and dataset
def run_benchmark(report_id, tool, tests, buildings, args):
    """Run benchmark with the given parameters."""

    # Convert the tool name to the corresponding tool object
    match tool:
        case "Evidently":
            tool = Evidently("Evidently", showReport=args.show_report)
        case "NannyML":
            tool = NannyML("NannyML", showReport=args.show_report)
        case "AlibiDetect":
            tool = AlibiDetect("AlibiDetect")

    # Convert the dataset name to the corresponding dataset object
    match args.dataset:
        case "energy":
            dataset = Data_Energy(f"{config.DATA_PATH}/energy_data.csv")
        case "occupancy":
            dataset = Data_Occupancy(f"{config.DATA_PATH}/occupancy_data.csv")

    # Run benchmark for each tool
    benchmark = Benchmark(tool, dataset, tests, buildings, args.vm)
    benchmark.runBenchmark(report_id)


# Run main function if the script is executed
if __name__ == "__main__":
    arguments = parser.parse_args()
    main(arguments)
