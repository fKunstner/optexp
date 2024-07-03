from typing import Literal, Optional

from optexp import config
from optexp.components.component import dataclass_component
from optexp.components.hardwareconfigs.hardwareconfig import (
    DetailedExpConfig,
    HardwareConfig,
)
from optexp.components.problem import Problem


@dataclass_component()
class RawHardwareConfig(HardwareConfig):

    num_workers: int
    micro_batch_size: int
    eval_micro_batch_size: Optional[int] = None
    device: Literal["cpu", "cuda", "auto"] = "auto"
    wandb_autosync: bool = False

    def load(self, problem: Problem) -> "RawDetailedExpConfig":
        return RawDetailedExpConfig(self, problem)

    def get_num_workers(self):
        return self.num_workers

    def get_accelerator(self) -> Literal["cpu", "cuda"]:
        match self.device:
            case "auto":
                return config.get_device()
            case "cpu":
                return "cpu"
            case "cuda":
                return "cuda"
            case _:
                raise ValueError(f"Unknown device {self.device}")

    def use_wandb_autosync(self) -> bool:
        return self.wandb_autosync


class RawDetailedExpConfig(DetailedExpConfig):

    def __init__(
        self,
        hardware_config: RawHardwareConfig,
        problem: Problem,
    ):
        tr_mbs, va_mbs = self._validate_batch_sizes(hardware_config, problem)
        self.tr_mbs: int = tr_mbs
        self.va_mbs: int = va_mbs
        self.acc = problem.batch_size // (tr_mbs * hardware_config.num_workers)
        self.exp_config: RawHardwareConfig = hardware_config

    def get_micro_batchsize_for_training(self) -> int:
        return self.tr_mbs

    def get_micro_batchsize_for_validation(self) -> int:
        return self.va_mbs

    @staticmethod
    def _validate_batch_sizes(hardware_config: RawHardwareConfig, problem: Problem):
        n_tr = problem.dataset.get_num_samples("tr")
        n_va = problem.dataset.get_num_samples("va")

        eff_bs = problem.batch_size
        w = hardware_config.num_workers
        tr_mbs = hardware_config.micro_batch_size
        va_mbs = (
            hardware_config.eval_micro_batch_size
            if hardware_config.eval_micro_batch_size is not None
            else tr_mbs
        )

        if n_tr % eff_bs != 0:
            raise ValueError(
                "Error in the batch size for training dataloader."
                "Batch size must divide number of training samples."
                f"Got batch size: {problem.batch_size},"
                f"number of training samples: {n_tr}"
            )

        if eff_bs % (tr_mbs * w) != 0:
            raise ValueError(
                "Batch size must be a multiple of micro batch size * num workers"
            )

        if n_va % (va_mbs * w) != 0:
            raise ValueError(
                "Error in the micro batch size for evaluation dataloader."
                "Num workers * Micro batch size must divide number of validation samples."
                f"Got micro batch size: {va_mbs},"
                f"Got num_workers: {w},"
                f"number of validation samples: {n_va}"
            )

        return tr_mbs, va_mbs

    def get_gradient_accumulation_steps(self) -> int:
        return self.acc
