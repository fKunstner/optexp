from abc import ABC, abstractmethod
from typing import Literal

from attr import frozen

from optexp.component import Component
from optexp.problem import Problem


@frozen
class BatchSizeInfo(Component):
    mbatchsize_tr: int
    mbatchsize_va: int
    accumulation_steps: int
    workers_tr: int
    workers_va: int


@frozen
class HardwareConfig(Component, ABC):
    """Abstract base class for hardware configurations."""

    @abstractmethod
    def get_batch_size_info(self, problem: Problem) -> BatchSizeInfo:
        raise NotImplementedError

    @abstractmethod
    def get_num_devices(self):
        raise NotImplementedError

    @abstractmethod
    def get_accelerator(self) -> Literal["cpu", "cuda"]:
        raise NotImplementedError
