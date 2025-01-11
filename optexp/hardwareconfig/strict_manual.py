from attrs import frozen

from optexp.hardwareconfig.hardwareconfig import BatchSizeInfo
from optexp.hardwareconfig.manual import ManualConfig
from optexp.hardwareconfig.utils import batchsize_mismatch_message
from optexp.problem import Problem


@frozen
class StrictManualConfig(ManualConfig):
    """Manual configuration for hardware settings.

    If you want to use multiple devices or if the batch size is too large to fit in memory,
    use this class to specify the hardware settings.

    The _effective_ batch size, the number of samples used to compute each optimization step,
    is specified in :class:`~optexp.Problem`.
    This class specifies the number of samples loaded at once, the micro batch size,
    which are then used with gradient accumulation to compute the optimization step.

    Operations like `drop_last` are not yet supported.
    You will get an error if the batch size does not divide the dataset size.

    Args:
        num_devices (int, optional): number of devices (eg GPUs). Defaults to 1.
        micro_batch_size (int, optional): mumber of samples loaded at once during training.
            Needs to evenly divide the batch size.
            If not provided, the :py:class:`Problem` batch size is used.
        eval_micro_batch_size (int, optional): number of samples loaded at once during evaluation.
            Size of the actual minibatches that will be loaded during evaluation.
            If not provided, the `micro_batch_size` is used.
        num_workers (int, optional): number of workers to load samples. Defaults to 0
        device (Literal["cpu" | "cuda" | "auto"]): device to use for training.
            Can be "cpu", "cuda" or "auto". Defaults to "auto", using the GPU if available.

    Example:

        1. Steps with a batch size of 100, loading 10 samples at a time::

            Problem(
                batch_size=100,
                hw_config=StrictManualConfig(
                    num_devices=1,
                    micro_batch_size=10
                ),
                ...,
            )

        2. Steps with a batch size of 100, loading 10 samples at a time with 2 GPUs::

            Problem(
                batch_size=100,
                hw_config=StrictManualConfig(
                    num_devices=2,
                    micro_batch_size=10,
                    device="gpu",
                ),
                ...,
            )

        3. Invalid configuration:
           The batch size is not a multiple of the micro batch size::

            Problem(
                batch_size=100,
                hw_config=StrictManualConfig(
                    num_devices=1,
                    micro_batch_size=15
                ),
                ...,
            )

        4. Invalid configuration:
           The ``batch_size`` is not a multiple of the ``micro_batch_size * num_devices``::

            Problem(
                batch_size=100,
                hw_config=StrictManualConfig(
                    num_devices=3,
                    micro_batch_size=50
                ),
                ...,
            )
    """

    def get_batch_size_info(self, problem: Problem) -> BatchSizeInfo:
        n_tr = problem.dataset.get_num_samples("tr")
        n_va = problem.dataset.get_num_samples("va")

        effective_bs = problem.batch_size
        tr_mbs = (
            self.micro_batch_size
            if self.micro_batch_size is not None
            else effective_bs // self.num_devices
        )
        va_mbs = (
            self.eval_micro_batch_size
            if self.eval_micro_batch_size is not None
            else tr_mbs
        )

        if n_tr % effective_bs != 0:
            raise ValueError(batchsize_mismatch_message("tr", n_tr, problem))

        if effective_bs % (tr_mbs * self.num_devices) != 0:
            raise ValueError(
                "Batch size must be a multiple of micro batch size * num devices. "
                f"Got batch size : {effective_bs}, "
                f"micro batch size: {tr_mbs}, num devices: {self.num_devices} (total: {self.num_devices * tr_mbs})"
            )

        if n_va % (va_mbs * self.num_devices) != 0:
            raise ValueError(
                "Error in the micro batch size for evaluation dataloader. "
                "Num devices * Micro batch size must divide number of validation samples. "
                f"Got micro batch size: {va_mbs}, "
                f"Got num_devices: {self.num_devices}, "
                f"number of validation samples: {n_va}"
            )

        return BatchSizeInfo(
            mbatchsize_tr=tr_mbs,
            mbatchsize_va=va_mbs,
            accumulation_steps=effective_bs // (tr_mbs * self.num_devices),
            workers_tr=self.num_workers,
            workers_va=self.num_workers,
        )
