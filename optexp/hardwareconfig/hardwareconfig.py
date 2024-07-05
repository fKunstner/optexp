from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

from optexp.component import Component
from optexp.problem import Problem


@dataclass(frozen=True)
class HardwareConfig(Component, ABC):
    """Abstract base class for hardware configurations."""

    @abstractmethod
    def load(self, problem: Problem) -> "_HardwareConfig":
        raise NotImplementedError

    @abstractmethod
    def get_num_devices(self):
        raise NotImplementedError

    @abstractmethod
    def get_accelerator(self) -> Literal["cpu", "cuda"]:
        raise NotImplementedError


class _HardwareConfig(ABC):
    """Abstract base class for the result of hardware configurations."""

    @abstractmethod
    def get_micro_batchsize_for_training(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_micro_batchsize_for_validation(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_gradient_accumulation_steps(self) -> int:
        raise NotImplementedError
