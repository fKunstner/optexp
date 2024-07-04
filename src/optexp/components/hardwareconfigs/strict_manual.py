from dataclasses import dataclass
from typing import Literal, Optional

from optexp import config
from optexp.components.hardwareconfig import HardwareConfig, ImplementationDetails
from optexp.components.hardwareconfigs.utils import batchsize_mismatch_message
from optexp.components.problem import Problem


@dataclass(frozen=True)
class StrictManualDetails(ImplementationDetails):

    num_workers: int
    micro_batch_size: int
    eval_micro_batch_size: Optional[int] = None
    device: Literal["cpu", "cuda", "auto"] = "auto"

    def load(self, problem: Problem) -> "StrictManualConfig":
        return StrictManualConfig(self, problem)

    def get_num_workers(self):
        return self.num_workers

    def get_accelerator(self) -> Literal["cpu", "cuda"]:
        if self.device == "auto":
            return config.get_device()
        if self.device in ["cpu", "cuda"]:
            return self.device  # type: ignore
        raise ValueError(f"Unknown device {self.device}")


class StrictManualConfig(HardwareConfig):

    def __init__(self, manual_details: "StrictManualDetails", problem: Problem):
        tr_mbs, va_mbs = self._validate_batch_sizes(manual_details, problem)
        self.tr_mbs: int = tr_mbs
        self.va_mbs: int = va_mbs
        self.acc = problem.batch_size // (tr_mbs * manual_details.num_workers)

    def get_micro_batchsize_for_training(self) -> int:
        return self.tr_mbs

    def get_micro_batchsize_for_validation(self) -> int:
        return self.va_mbs

    def get_gradient_accumulation_steps(self) -> int:
        return self.acc

    @staticmethod
    def _validate_batch_sizes(manual_details: "StrictManualDetails", problem: Problem):
        n_tr = problem.dataset.get_num_samples("tr")
        n_va = problem.dataset.get_num_samples("va")

        effective_bs = problem.batch_size
        w = manual_details.num_workers
        tr_mbs = manual_details.micro_batch_size
        va_mbs = (
            manual_details.eval_micro_batch_size
            if manual_details.eval_micro_batch_size is not None
            else tr_mbs
        )

        if n_tr % effective_bs != 0:
            raise ValueError(batchsize_mismatch_message("tr", n_tr, problem))

        if effective_bs % (tr_mbs * w) != 0:
            raise ValueError(
                "Batch size must be a multiple of micro batch size * num workers."
                f"Got batch size : {effective_bs}, "
                f"micro batch size: {tr_mbs}, num workers: {w} (total: {w * tr_mbs})"
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
