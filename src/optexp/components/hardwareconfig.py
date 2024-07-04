from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

from optexp.components.component import Component
from optexp.components.problem import Problem


@dataclass(frozen=True)
class ImplementationDetails(Component, ABC):

    @abstractmethod
    def load(self, problem: Problem) -> "HardwareConfig":
        raise NotImplementedError

    @abstractmethod
    def get_num_workers(self):
        raise NotImplementedError

    @abstractmethod
    def get_accelerator(self) -> Literal["cpu", "cuda"]:
        raise NotImplementedError


class HardwareConfig(ABC):

    @abstractmethod
    def get_micro_batchsize_for_training(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_micro_batchsize_for_validation(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_gradient_accumulation_steps(self) -> int:
        raise NotImplementedError
