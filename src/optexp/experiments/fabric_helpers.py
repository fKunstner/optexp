from typing import Dict

from torch import Tensor


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
