"""Some docstring here"""

from d3bench.tools import Tool
from d3bench.benchmarks import Benchmark
from d3bench.config import Data


class Category:
    """Base class for the categories."""


class OnlineCD(Category):
    def __init__(self, tool: Tool, data: Data) -> None:
        self.methods = tool.online_cd_methods
        self.benchmarks = [
            Benchmark(tool, method, , settings=None) for method in self
        ]



    def benchmarks(self, data) -> list[Benchmark]:

        self.benchmarks = [
            Benchmark(tool, method, *data, settings=None) for method in self.methods
        ]


class OnlineDD(Category):
    def __init__(self, tool: Tool) -> None:
        self.methods = tool.online_dd_methods


class BatchCD(Category):
    def __init__(self, tool: Tool) -> None:
        self.methods = tool.batch_cd_methods


class BatchDD(Category):
    def __init__(self, tool: Tool) -> None:
        self.methods = tool.batch_dd_methods
