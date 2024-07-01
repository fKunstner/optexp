import math
import pprint
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import lightning as ptl
import numpy as np
import torch
import torch.nn
from torch import Tensor
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import DataLoader

from optexp.datasets.dataset import TrVa
from optexp.experiments.experiment import Experiment
from optexp.experiments.hardwareconfig import DetailedExpConfig
from optexp.loggers import DataLogger
from optexp.loggers.asdict_with_classes import asdict_with_class
from optexp.problems.metrics import Metric
from optexp.runner.fabric_helpers import (
    TrainMode,
    loginfo_on_r0,
    synchronised_log,
    EvalMode,
)
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
                run(exp)
        prof.export_chrome_trace("trace.json")
    else:
        run(exp)


@dataclass
class TimeCounter:
    epoch: int = 0
    step: int = 0
    step_within_epoch: int = 0

    def next_iter(self):
        self.step += 1
        self.step_within_epoch += 1

    def next_epoch(self):
        self.epoch += 1
        self.step_within_epoch = 0


@dataclass
class DataLoaders:
    tr_tr: DataLoader
    tr_va: DataLoader
    va_va: DataLoader


@dataclass
class ExperimentState:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    dataloaders: DataLoaders
    metrics: Iterable[Metric]
    loss_func: torch.nn.Module
    _tr_dl_iter = None
    time: TimeCounter = TimeCounter()

    def get_batch(self):
        if self._tr_dl_iter is None:
            self._tr_dl_iter = iter(self.dataloaders.tr_tr)
            self.time = TimeCounter()

        try:
            features, labels = next(self._tr_dl_iter)
        except StopIteration:
            self.time.next_epoch()
            tr_iterator = iter(self.dataloaders.tr_tr)
            features, labels = next(tr_iterator)
        finally:
            self.time.next_iter()

        return features, labels


def run(exp: Experiment) -> None:

    fabric = ptl.Fabric(
        accelerator=exp.hw_config.get_accelerator(),
        devices=exp.hw_config.get_num_workers(),
        num_nodes=1,
        strategy="auto",
    )
    fabric.launch()
    # only rank 0 gets a real one

    data_logger: Optional[DataLogger] = None
    if fabric.global_rank == 0:
        data_logger = DataLogger(
            config_dict=asdict_with_class(exp),
            group=exp.group,
            run_id=time.strftime("%Y-%m-%d--%H-%M-%S"),
            exp_id=exp.exp_id(),
            save_directory=exp.save_directory(),
            wandb_autosync=exp.hw_config.use_wandb_autosync(),
        )
    fabric.barrier()

    loginfo_on_r0(fabric, "=" * 80)
    loginfo_on_r0(fabric, "Initializing experiment:")
    loginfo_on_r0(fabric, pprint.pformat(asdict_with_class(exp), indent=4))
    loginfo_on_r0(fabric, "=" * 80)
    exp_state, detailed_exp_config = initialize(exp, fabric)

    loginfo_on_r0(fabric, "Initial evaluation...")
    eval_and_log(fabric, exp_state, {}, data_logger)

    loginfo_on_r0(fabric, "Starting training...")
    for t in range(1, exp.steps + 1):
        live_loss, exp_state = training_step(fabric, exp_state, detailed_exp_config)

        if math.isnan(live_loss) or math.isinf(live_loss):
            break

        if t % exp.eval_every == 0:
            extra_dict = {
                "live_loss": live_loss,
            }
            eval_and_log(fabric, exp_state, extra_dict, data_logger)

    if fabric.global_rank == 0 and data_logger is not None:
        data_logger.finish(exit_code=0)

    fabric.barrier()


def initialize(
    exp: Experiment, fabric: ptl.Fabric
) -> Tuple[ExperimentState, DetailedExpConfig]:

    loginfo_on_r0(fabric, "Initializing problem configuration")
    hw_config = exp.hw_config.load(exp.problem)

    seed = exp.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    loginfo_on_r0(fabric, "Loading dataset...")
    tr_tr_dl = exp.problem.dataset.get_dataloader(
        tr_va="tr", b=hw_config.get_micro_batchsize_for_training()
    )
    tr_va_dl = exp.problem.dataset.get_dataloader(
        tr_va="tr", b=hw_config.get_micro_batchsize_for_validation()
    )
    va_va_dl = exp.problem.dataset.get_dataloader(
        tr_va="va", b=hw_config.get_micro_batchsize_for_validation()
    )

    loginfo_on_r0(fabric, "Loading model...")
    model = exp.problem.model.load_model(
        exp.problem.dataset.input_shape(hw_config.get_micro_batchsize_for_training()),
        exp.problem.dataset.output_shape(hw_config.get_micro_batchsize_for_training()),
    )

    loginfo_on_r0(fabric, "Loading loss function, metrics, and optimizers...")
    loss_func = exp.problem.lossfunc()
    metrics = [metric() for metric in exp.problem.metrics]
    opt = exp.optim.load(model)
    model, opt = fabric.setup(model, opt)

    return (
        ExperimentState(
            model=model,
            optimizer=opt,
            dataloaders=DataLoaders(
                *fabric.setup_dataloaders(
                    tr_tr_dl, tr_va_dl, va_va_dl, move_to_device=True
                )
            ),
            metrics=metrics,
            loss_func=loss_func,
        ),
        hw_config,
    )


def eval_and_log(
    fabric: ptl.Fabric,
    exp_data: ExperimentState,
    extra_dict: Dict[str, Any],
    data_logger: Optional[DataLogger] = None,
) -> None:

    time_dict = {
        "epoch": exp_data.time.epoch,
        "step": exp_data.time.step,
        "step_within_epoch": exp_data.time.step_within_epoch,
    }

    metrics_tr = evaluate(
        loader=exp_data.dataloaders.tr_va,
        metrics=exp_data.metrics,
        model=exp_data.model,
    )
    metrics_va = evaluate(
        loader=exp_data.dataloaders.va_va,
        metrics=exp_data.metrics,
        model=exp_data.model,
    )
    reduced_metrics_tr = reduce_and_make_dictionary(fabric, metrics_tr, "tr")
    reduced_metrics_va = reduce_and_make_dictionary(fabric, metrics_va, "va")

    synchronised_log(
        fabric,
        data_logger,
        reduced_metrics_tr,
        reduced_metrics_va,
        time_dict,
        extra_dict,
    )
    torch.cuda.empty_cache()


def evaluate(
    loader: DataLoader,
    metrics: Iterable[Metric],
    model: torch.nn.Module,
) -> Dict[Metric, SumAndCounter]:
    running_metrics: Dict[Metric, SumAndCounter] = {
        metric: SumAndCounter(torch.tensor(0.0), torch.tensor(0.0))
        for metric in metrics
    }

    with EvalMode(model), torch.no_grad():
        for _, (features, labels) in enumerate(loader):
            y_pred = model(features)
            b = len(labels)

            for metric in metrics:
                loss, weight = to_loss_and_weight(metric(y_pred, labels), b)
                running_metrics[metric] += SumAndCounter(loss.detach(), weight.detach())

        return running_metrics


def training_step(
    fabric: ptl.Fabric,
    exp_state: ExperimentState,
    detailed_exp_config: DetailedExpConfig,
) -> Tuple[float, ExperimentState]:

    with TrainMode(exp_state.model):
        exp_state.optimizer.zero_grad()
        total_loss_and_count = SumAndCounter(torch.tensor(0.0), torch.tensor(0.0))
        for _ in range(detailed_exp_config.get_gradient_accumulation_steps()):
            features, labels = exp_state.get_batch()
            b = len(labels)
            loss, weight = to_loss_and_weight(
                exp_state.loss_func(exp_state.model(features), labels), b
            )
            total_loss_and_count += SumAndCounter(loss.detach(), weight)

            fabric.backward(loss)

        for p in exp_state.model.parameters():
            if p.grad is not None:
                p.grad /= total_loss_and_count.denominator

        train_loss = total_loss_and_count.reduce(fabric).cpu().item()
        exp_state.optimizer.step()
        torch.cuda.empty_cache()

        return train_loss, exp_state


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
