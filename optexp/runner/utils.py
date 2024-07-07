import time
from dataclasses import dataclass
from typing import Optional, Tuple

import lightning as ptl
import torch
import torch.nn
from torch import Tensor

from optexp.config import get_logger


def reduce_tensor(fabric, val: Tensor, reduce_op: str = "sum") -> Tensor:
    return fabric.all_reduce(val, reduce_op=reduce_op)  # type: ignore[arg-type]


@dataclass
class SumAndCounter:
    numerator: Tensor
    denominator: Tensor

    @staticmethod
    def zero() -> "SumAndCounter":
        return SumAndCounter(torch.tensor(0.0), torch.tensor(0.0))

    def __add__(self, other: "SumAndCounter") -> "SumAndCounter":
        return SumAndCounter(
            self.numerator + other.numerator, self.denominator + other.denominator
        )

    def reduce(self, fabric: ptl.Fabric, reduce_op="sum") -> tuple[Tensor, Tensor]:
        numerator = reduce_tensor(fabric, self.numerator, reduce_op=reduce_op)
        denominator = reduce_tensor(fabric, self.denominator, reduce_op=reduce_op)
        return numerator, denominator

    def result(self) -> Tensor:
        return self.numerator / self.denominator

    @staticmethod
    def from_output(output: Tensor | Tuple[Tensor, Tensor]):
        if isinstance(output, tuple) and len(output) == 2:
            return SumAndCounter(output[0], output[1])
        return SumAndCounter(output, torch.tensor(1))


class RateLimiter:
    last_message_time: Optional[float] = None
    RATE_LIMIT = 1


def should_print(rate_limited: bool, last_message_timestamp: Optional[float]):
    if not rate_limited:
        return True

    if last_message_timestamp is None:
        return True

    time_since_last = time.time() - last_message_timestamp
    if time_since_last > rate_limited:
        return True

    return False


def loginfo_on_r0(fabric, message: str, rate_limited: bool = False) -> None:
    if fabric.global_rank == 0:
        if should_print(rate_limited, RateLimiter.last_message_time):
            RateLimiter.last_message_time = time.time()
            get_logger().info(message)


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
