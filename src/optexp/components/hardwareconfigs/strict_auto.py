from dataclasses import dataclass
from typing import Literal

from optexp import config
from optexp.components.hardwareconfig import HardwareConfig, ImplementationDetails
from optexp.components.hardwareconfigs.utils import batchsize_mismatch_message
from optexp.components.problem import Problem


@dataclass(frozen=True)
class StrictAutoDetails(ImplementationDetails):
    def load(self, problem: Problem) -> "StrictAutoConfig":
        return StrictAutoConfig(self, problem)

    def get_num_workers(self):
        return 1

    def get_accelerator(self) -> Literal["cpu", "cuda"]:
        return config.get_device()


class StrictAutoConfig(HardwareConfig):

    def __init__(self, auto_details: "StrictAutoDetails", problem: Problem):
        self._validate_batch_sizes(auto_details, problem)
        self.tr_mbs: int = problem.batch_size
        self.va_mbs: int = problem.batch_size
        self.acc = 1

    def get_micro_batchsize_for_training(self) -> int:
        return self.tr_mbs

    def get_micro_batchsize_for_validation(self) -> int:
        return self.va_mbs

    @staticmethod
    def _validate_batch_sizes(
        auto_details: "StrictAutoDetails", problem: Problem
    ):  # pylint: disable=unused-argument

        n_tr = problem.dataset.get_num_samples("tr")
        n_va = problem.dataset.get_num_samples("va")

        if n_tr % problem.batch_size != 0:
            raise ValueError(batchsize_mismatch_message("tr", problem.batch_size, n_tr))

        if n_va % problem.batch_size != 0:
            raise ValueError(batchsize_mismatch_message("va", problem.batch_size, n_va))

    def get_gradient_accumulation_steps(self) -> int:
        return self.acc
