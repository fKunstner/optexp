import math
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import lightning as ptl
import torch
import torch.nn
from torch import Tensor
from tqdm import tqdm as _tqdm

from optexp.config import Config, get_logger
from optexp.experiment import Experiment
from optexp.metrics import LossLikeMetric
from optexp.optim.optimizer import Regularizable
from optexp.results.data_logger import DummyDataLogger
from optexp.results.main_data_logger import MainDataLogger


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

    def reduce_and_divide(
        self, fabric: ptl.Fabric, reduce_op="sum"
    ) -> List[float] | float:
        numerator, denominator = self.reduce(fabric, reduce_op=reduce_op)
        v = (numerator / denominator).cpu()
        return v.tolist() if torch.numel(v) > 1 else v.item()

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


def tqdm(*args, **kwargs):
    if Config.tqdm_enabled:
        return _tqdm(*args, **kwargs)
    return args[0]


def make_datalogger(exp, fabric):
    if fabric.global_rank == 0:
        return MainDataLogger(experiment=exp)
    return DummyDataLogger()


def get_losslike_metrics(exp: Experiment) -> list[LossLikeMetric]:
    return [
        metric for metric in exp.problem.metrics if isinstance(metric, LossLikeMetric)
    ]


def has_losslike_metrics_to_evaluate(exp: Experiment) -> bool:
    return len(get_losslike_metrics(exp)) > 0


def current_time(exp_state):
    return {
        "epoch": exp_state.iteration_counter.epoch,
        "step": exp_state.iteration_counter.steps,
        "step_within_epoch": exp_state.iteration_counter.steps_within_epoch,
        "mb": exp_state.iteration_counter.microbatches,
        "mb_within_epoch": exp_state.iteration_counter.microbatches_within_epoch,
    }


def regularization(exp, exp_state):
    if isinstance(exp.optim, Regularizable):
        return {
            "regularization": exp.optim.regularizer_loss(exp_state.model).cpu().item()
        }
    return {}


def should_early_stop(fabric: ptl.Fabric, live_loss: float):
    is_diverging = math.isnan(live_loss) or math.isinf(live_loss)
    synced_is_diverging = fabric.strategy.reduce_boolean_decision(
        is_diverging, all=False
    )
    return synced_is_diverging
