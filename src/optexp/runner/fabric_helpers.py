from typing import Any, Dict

import torch.nn
from torch import Tensor

from optexp.config import get_logger
from optexp.loggers import DataLogger


def reduce(fabric, val: float, reduce_op="sum") -> Tensor:
    return fabric.all_reduce(val, reduce_op=reduce_op)  # type: ignore[arg-type]


def reduce_tensor(fabric, val: Tensor, reduce_op: str = "sum") -> Tensor:
    return fabric.all_reduce(val, reduce_op=reduce_op)  # type: ignore[arg-type]


def gather(fabric, val: float) -> Tensor:
    return fabric.all_gather(val)  # type: ignore[arg-type]


def gather_dict(fabric, val: Dict[str, Tensor]) -> Dict[str, Tensor]:
    return fabric.all_gather(val)  # type: ignore[arg-type]


def gather_bool(fabric, exceptions: Dict[str, bool]) -> Dict[str, Tensor]:
    return fabric.all_gather(exceptions)  # type: ignore[arg-type]


def tensor_any(param: Tensor):
    return any(param) if len(param.shape) > 0 else bool(param)


def loginfo_on_r0(fabric, message: str) -> None:
    if fabric.global_rank == 0:
        get_logger().info(message)


def synchronised_log(
    fabric,
    data_logger: DataLogger | None,
    *dictionaries_to_log: Dict[str, Any],
) -> None:
    if fabric.global_rank == 0 and data_logger is not None:
        for dict_to_log in dictionaries_to_log:
            data_logger.log_data(dict_to_log)
        data_logger.commit()
    fabric.barrier()


class EvalMode:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self._original_train_mode = model.training

    def __enter__(self):
        self.model.eval()
        return self.model

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self._original_train_mode:
            self.model.train()
        else:
            self.model.eval()


class TrainMode:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self._original_train_mode = model.training

    def __enter__(self):
        self.model.train()
        return self.model

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self._original_train_mode:
            self.model.train()
        else:
            self.model.eval()
