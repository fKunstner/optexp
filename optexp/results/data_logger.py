from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from optexp.experiment import Experiment


class DataLogger(ABC):
    @abstractmethod
    def log(self, metric_dict: dict[str, float | list[float]]) -> None:
        pass

    @abstractmethod
    def commit(self) -> None:
        pass

    @abstractmethod
    def finish(self, exit_code, stopped_early) -> None:
        pass


class DummyDataLogger(DataLogger):
    """A dummy results logger that does nothing."""

    def __init__(self, experiment: Optional[Experiment] = None) -> None:
        pass

    def log(self, metric_dict: dict) -> None:
        pass

    def commit(self) -> None:
        pass

    def finish(self, exit_code, stopped_early) -> None:
        pass
