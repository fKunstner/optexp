from typing import Literal, Optional

from attrs import frozen

from optexp.config import Config
from optexp.hardwareconfig.hardwareconfig import BatchSizeInfo, HardwareConfig
from optexp.problem import Problem


@frozen
class ManualConfig(HardwareConfig):
    """Manual configuration for hardware settings.
    Does not ensure deterministic reproducibility! Use StrictManualConfig.

    The _effective_ batch size, the number of samples used to compute each optimization step,
    is specified in :class:`~optexp.Problem`.
    This class specifies the number of samples loaded at once, the micro batch size,
    which are then used with gradient accumulation to compute the optimization step.

    You will get an error if the batch size does not divide the dataset size.

    Args:
        num_devices (int, optional): number of devices (eg GPUs). Defaults to 1.
        micro_batch_size (int, optional): mumber of samples loaded at once during training.
            If not provided, the :py:class:`Problem` batch size is used.
        eval_micro_batch_size (int, optional): number of samples loaded at once during evaluation.
            Size of the actual minibatches that will be loaded during evaluation.
            If not provided, the `micro_batch_size` is used.
        num_workers (int, optional): number of workers to load samples. Defaults to 0
        device (Literal["cpu" | "cuda" | "auto"]): device to use for training.
            Can be "cpu", "cuda" or "auto". Defaults to "auto", using the GPU if available.
    """

    num_devices: int = 1
    micro_batch_size: Optional[int] = None
    eval_micro_batch_size: Optional[int] = None
    num_workers: int = 0
    device: Literal["cpu", "cuda", "auto"] = "auto"

    def get_num_devices(self):
        return self.num_devices

    def get_accelerator(self) -> Literal["cpu", "cuda"]:
        if self.device == "auto":
            return Config.get_device()
        if self.device in ["cpu", "cuda"]:
            return self.device  # type: ignore
        raise ValueError(f"Unknown device {self.device}")

    def get_batch_size_info(self, problem: Problem) -> BatchSizeInfo:
        effective_bs = problem.batch_size
        num_workers = self.num_devices
        tr_mbs = (
            self.micro_batch_size
            if self.micro_batch_size is not None
            else effective_bs // num_workers
        )
        va_mbs = (
            self.eval_micro_batch_size
            if self.eval_micro_batch_size is not None
            else tr_mbs
        )

        return BatchSizeInfo(
            mbatchsize_tr=tr_mbs,
            mbatchsize_va=va_mbs,
            accumulation_steps=effective_bs // (tr_mbs * num_workers),
            workers_tr=self.num_workers,
            workers_va=self.num_workers,
        )
