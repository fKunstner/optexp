import math
import pprint
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Literal, Optional, Tuple

import lightning as ptl
import numpy as np
import torch
import torch.nn
from torch import Tensor
from torch.profiler import ProfilerActivity, profile, record_function

from optexp.config import get_device, get_logger
from optexp.datasets.dataset import TrVa
from optexp.experiments.experiment import Experiment
from optexp.loggers import DataLogger
from optexp.loggers.asdict_with_classes import asdict_with_class
from optexp.problems import DivergingException
from optexp.problems.metrics import Metric
from optexp.runner.fabric_exception_helpers import (
    SynchronizedDivergence,
    SynchronizedError,
    sync_try_except,
)
from optexp.runner.fabric_helpers import info_r0, synchronised_log
from optexp.runner.sum_and_counter import SumAndCounter


def run_experiment(exp: Experiment, run_profiler: bool = False) -> None:
    """Run the experiment.

    Initializes the problem and optimizer and optimizes the
    problem given the optimizer for the defined amount of epochs.
    Logs the loss function values/metrics returned during the eval and training.
    Catches any exception raised during this process and logs it before exiting.

    Raises:
        BaseException: Raised when user Ctrl+C when experiments is running.
    """

    if run_profiler:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            with record_function("main_run"):
                fabric, data_logger = initialize_fabric_and_data_logger(exp, DataLogger)
                exception_managed_run(fabric, exp, data_logger)
        prof.export_chrome_trace("trace.json")
    else:
        fabric, data_logger = initialize_fabric_and_data_logger(exp, DataLogger)
        exception_managed_run(fabric, exp, data_logger)


def exception_managed_run(
    fabric: ptl.Fabric,
    exp: Experiment,
    data_logger: Optional[DataLogger],
):
    experiment_status: Optional[Literal["finished", "diverged", "error"]] = None
    try:
        run(fabric, exp, data_logger)
    except SynchronizedDivergence:
        experiment_status = "diverged"
        raise
    except SynchronizedError:
        experiment_status = "error"
        raise
    else:
        experiment_status = "finished"
    finally:
        if fabric.global_rank == 0:
            get_logger().warning("TERMINATING.")

            match experiment_status:
                case "finished":
                    exit_code = 0
                case "diverged":
                    exit_code = 0
                case "error":
                    exit_code = 1
                case None:
                    exit_code = 2
                case _:
                    exit_code = 3

            if data_logger is not None:
                data_logger.save(exit_code=exit_code)


def initialize_fabric_and_data_logger(
    exp, data_logger_class
) -> Tuple[ptl.Fabric, Optional[DataLogger]]:
    fabric = ptl.Fabric(
        accelerator=get_device(),
        devices=exp.devices,
        num_nodes=exp.nodes,
        strategy=exp.strategy,
    )
    fabric.launch()
    # only rank 0 gets a real one
    data_logger = None
    if fabric.global_rank == 0:
        data_logger = data_logger_class(
            config_dict=asdict_with_class(exp),
            group=exp.group,
            run_id=time.strftime("%Y-%m-%d--%H-%M-%S"),
            exp_id=exp.exp_id(),
            save_directory=exp.save_directory(),
            wandb_autosync=exp.wandb_autosync,
        )
    fabric.barrier()
    return fabric, data_logger


@dataclass
class ExperimentState:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    tr_dataloader: torch.utils.data.DataLoader
    va_dataloader: torch.utils.data.DataLoader
    metrics: Iterable[Metric]
    loss_func: torch.nn.Module
    _tr_dl_iter = None

    def get_batch(self):
        if self._tr_dl_iter is None:
            self._tr_dl_iter = iter(self.tr_dataloader)

        try:
            features, labels = next(self._tr_dl_iter)
        except StopIteration:
            tr_iterator = iter(self.tr_dataloader)
            features, labels = next(tr_iterator)
        return features, labels


def run(
    fabric: ptl.Fabric, exp: Experiment, data_logger: Optional[DataLogger] = None
) -> None:

    info_r0(fabric, "=" * 80)
    info_r0(fabric, "Initializing experiment:")
    info_r0(fabric, pprint.pformat(asdict_with_class(exp), indent=4))
    info_r0(fabric, "=" * 80)
    exp_state = initialize(exp, fabric)

    info_r0(fabric, "Initial evaluation...")
    eval_and_log(fabric, exp_state, {"step": 0}, data_logger)

    info_r0(fabric, "Starting training...")
    for t in range(1, exp.steps + 1):
        live_loss = training_step(fabric, exp, exp_state)
        if t % exp.eval_every == 0:
            extra_dict = {"step": t, "live_loss": live_loss}
            eval_and_log(fabric, exp_state, extra_dict, data_logger)


def initialize(exp: Experiment, fabric: ptl.Fabric) -> ExperimentState:

    apply_seed(exp)

    info_r0(fabric, "Loading dataset...")

    def get_dataloaders():
        b = exp.problem.batch_size
        return (
            exp.problem.dataset.input_shape(b),
            exp.problem.dataset.output_shape(b),
            exp.problem.dataset.get_dataloader(tr_va="tr", b=b),
            exp.problem.dataset.get_dataloader(tr_va="va", b=b),
        )

    input_shape, output_shape, tr_dl, va_dl = sync_try_except(fabric, get_dataloaders)
    info_r0(fabric, "Loading model...")

    def get_model():
        return exp.problem.model.load_model(input_shape, output_shape)

    model = sync_try_except(fabric, get_model)

    info_r0(fabric, "Loading loss function, metrics, and optimizers...")

    def get_loss_metrics_and_optimizer():
        return (
            exp.problem.lossfunc(),
            [metric() for metric in exp.problem.metrics],
            exp.optim.load(model),
        )

    loss_func, metrics, opt = sync_try_except(fabric, get_loss_metrics_and_optimizer)

    info_r0(fabric, "Fabric setup...")
    model, opt = fabric.setup(model, opt)
    tr_dl, va_dl = fabric.setup_dataloaders(tr_dl, va_dl, move_to_device=True)

    return ExperimentState(
        model=model,
        optimizer=opt,
        tr_dataloader=tr_dl,
        va_dataloader=va_dl,
        metrics=metrics,
        loss_func=loss_func,
    )


def eval_and_log(
    fabric: ptl.Fabric,
    state: ExperimentState,
    extra_dict: Dict[str, Any],
    data_logger: Optional[DataLogger] = None,
) -> None:

    def compute_tr_metrics() -> Dict[Metric, SumAndCounter]:
        return evaluate(
            loader=state.tr_dataloader, metrics=state.metrics, model=state.model
        )

    def compute_va_metrics() -> Dict[Metric, SumAndCounter]:
        return evaluate(
            loader=state.va_dataloader, metrics=state.metrics, model=state.model
        )

    metrics_tr = sync_try_except(fabric, compute_tr_metrics)
    metrics_va = sync_try_except(fabric, compute_va_metrics)
    reduced_metrics_tr = reduce_and_make_dictionary(fabric, metrics_tr, "tr")
    reduced_metrics_va = reduce_and_make_dictionary(fabric, metrics_va, "va")

    synchronised_log(
        fabric,
        data_logger,
        reduced_metrics_tr,
        reduced_metrics_va,
        extra_dict,
    )
    torch.cuda.empty_cache()


def evaluate(
    loader: torch.utils.data.DataLoader,
    metrics: Iterable[Metric],
    model: torch.nn.Module,
) -> Dict[Metric, SumAndCounter]:
    running_metrics: Dict[Metric, SumAndCounter] = {
        metric: SumAndCounter(torch.tensor(0.0), torch.tensor(0.0))
        for metric in metrics
    }

    with torch.no_grad():
        for _, (features, labels) in enumerate(loader):
            y_pred = model(features)
            b = len(labels)

            for metric in metrics:
                loss, weight = to_loss_and_weight(metric(y_pred, labels), b)
                running_metrics[metric] += SumAndCounter(loss.detach(), weight.detach())

    return running_metrics


def training_step(
    fabric: ptl.Fabric, exp: Experiment, exp_state: ExperimentState
) -> float:
    def compute_loss():
        features, labels = exp_state.get_batch()
        b = len(labels)
        loss, weight = to_loss_and_weight(
            exp_state.loss_func(exp_state.model(features), labels), b
        )
        loss_and_count = SumAndCounter(loss.item(), weight)
        if math.isnan(loss) or math.isinf(loss):
            raise DivergingException()
        return loss, loss_and_count

    exp_state.optimizer.zero_grad()
    total_loss_and_count = SumAndCounter(torch.tensor(0.0), torch.tensor(0.0))
    for _ in range(exp.gradient_acc_steps):
        loss, current_loss_and_count = sync_try_except(fabric, compute_loss)
        total_loss_and_count += current_loss_and_count
        fabric.backward(loss)

    for p in exp_state.model.parameters():
        if p.grad is not None:
            p.grad /= total_loss_and_count.denominator

    train_loss = total_loss_and_count.reduce(fabric).cpu().item()
    exp_state.optimizer.step()
    torch.cuda.empty_cache()

    return train_loss


def reduce_and_make_dictionary(
    fabric: ptl.Fabric, metrics_and_counts_raw: Dict[Metric, SumAndCounter], tr_va: TrVa
) -> Dict[str, Iterable | float]:
    def reduce(x):
        v = x.reduce(fabric).cpu()
        return v.tolist() if torch.numel(v) > 1 else v.item()

    values_by_str: Dict[str, Iterable | float] = {
        f"{tr_va}_{str(k.__class__.__name__)}": reduce(v)
        for k, v in metrics_and_counts_raw.items()
    }
    return values_by_str


def to_loss_and_weight(
    output: Tensor | Tuple[Tensor, Tensor], default_weight: float
) -> Tuple[Tensor, Tensor]:
    if isinstance(output, tuple):
        if len(output) != 2:
            raise ValueError(
                "Invalid parameter. Got tuple of size != 2. "
                "Expected either Tensor or Tuple[Tensor, Tensor]"
            )
        loss, weight = output
        return loss, weight

    if isinstance(output, Tensor):
        loss = output
        loss *= default_weight
        return loss, torch.tensor(default_weight)

    raise ValueError(
        f"Invalid parameter. Got {type(output)}."
        "Expected either Tensor or Tuple[Tensor, Tensor]"
    )


def apply_seed(self) -> None:
    np.random.seed(self.seed)
    random.seed(self.seed)
    torch.manual_seed(self.seed)
    torch.cuda.manual_seed_all(self.seed)
