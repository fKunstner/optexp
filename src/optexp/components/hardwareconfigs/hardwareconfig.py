from abc import ABC, abstractmethod
from typing import Literal

from optexp.components.component import Component, dataclass_component
from optexp.components.problem import Problem


class DetailedExpConfig(ABC):

    @abstractmethod
    def get_micro_batchsize_for_training(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_micro_batchsize_for_validation(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_gradient_accumulation_steps(self) -> int:
        raise NotImplementedError


@dataclass_component()
class HardwareConfig(Component, ABC):

    @abstractmethod
    def load(self, problem: Problem) -> DetailedExpConfig:
        raise NotImplementedError

    @abstractmethod
    def get_num_workers(self):
        raise NotImplementedError

    @abstractmethod
    def get_accelerator(self) -> Literal["cpu", "cuda"]:
        raise NotImplementedError

    @abstractmethod
    def use_wandb_autosync(self) -> bool:
        raise NotImplementedError
