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

from optexp.config import get_device
from optexp.datasets.dataset import TrVa
from optexp.experiments.experiment import Experiment
from optexp.loggers import DataLogger
from optexp.loggers.asdict_with_classes import asdict_with_class
from optexp.problems.metrics import Metric
from optexp.runner.fabric_helpers import loginfo_on_r0, synchronised_log
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


def run(exp: Experiment) -> None:

    fabric = ptl.Fabric(
        accelerator=get_device(),
        devices=exp.devices,
        num_nodes=exp.nodes,
        strategy=exp.strategy,
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
            wandb_autosync=exp.wandb_autosync,
        )
    fabric.barrier()

    loginfo_on_r0(fabric, "=" * 80)
    loginfo_on_r0(fabric, "Initializing experiment:")
    loginfo_on_r0(fabric, pprint.pformat(asdict_with_class(exp), indent=4))
    loginfo_on_r0(fabric, "=" * 80)
    exp_state = initialize(exp, fabric)

    loginfo_on_r0(fabric, "Initial evaluation...")
    eval_and_log(fabric, exp_state, {"step": 0}, data_logger)

    loginfo_on_r0(fabric, "Starting training...")

    for t in range(1, exp.steps + 1):
        live_loss = training_step(fabric, exp, exp_state)

        if math.isnan(live_loss) or math.isinf(live_loss):
            break

        if t % exp.eval_every == 0:
            extra_dict = {"step": t, "live_loss": live_loss}
            eval_and_log(fabric, exp_state, extra_dict, data_logger)

    if fabric.global_rank == 0 and data_logger is not None:
        data_logger.finish(exit_code=0)

    fabric.barrier()


def initialize(exp: Experiment, fabric: ptl.Fabric) -> ExperimentState:

    np.random.seed(exp.seed)
    random.seed(exp.seed)
    torch.manual_seed(exp.seed)
    torch.cuda.manual_seed_all(exp.seed)

    loginfo_on_r0(fabric, "Loading dataset...")
    b = exp.problem.batch_size
    tr_dl = exp.problem.dataset.get_dataloader(tr_va="tr", b=b)
    va_dl = exp.problem.dataset.get_dataloader(tr_va="va", b=b)

    loginfo_on_r0(fabric, "Loading model...")
    input_shape = exp.problem.dataset.input_shape(b)
    output_shape = exp.problem.dataset.output_shape(b)
    model = exp.problem.model.load_model(input_shape, output_shape)

    loginfo_on_r0(fabric, "Loading loss function, metrics, and optimizers...")
    loss_func = exp.problem.lossfunc()
    metrics = [metric() for metric in exp.problem.metrics]
    opt = exp.optim.load(model)
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

    metrics_tr = evaluate(
        loader=state.tr_dataloader, metrics=state.metrics, model=state.model
    )
    metrics_va = evaluate(
        loader=state.va_dataloader, metrics=state.metrics, model=state.model
    )
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
    exp_state.optimizer.zero_grad()
    total_loss_and_count = SumAndCounter(torch.tensor(0.0), torch.tensor(0.0))
    for _ in range(exp.gradient_acc_steps):
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
