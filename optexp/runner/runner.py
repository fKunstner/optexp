import math
import pprint
import random
from typing import Any, Dict, Iterable, Tuple

import lightning as ptl
import numpy as np
import torch
import torch.nn
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import DataLoader

from optexp.data.data_logger import DataLogger, DummyDataLogger, WandbDataLogger
from optexp.experiment import Experiment
from optexp.metrics.metric import Metric
from optexp.runner.exp_state import DataLoaders, ExperimentState
from optexp.runner.utils import EvalMode, SumAndCounter, TrainMode, loginfo_on_r0


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
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            use_cuda=True,
        ) as prof:
            run(exp)
        prof.export_chrome_trace("trace.json")
    else:
        run(exp)


def should_early_stop(fabric: ptl.Fabric, live_loss: float):
    is_diverging = math.isnan(live_loss) or math.isinf(live_loss)
    synced_is_diverging = fabric.strategy.reduce_boolean_decision(
        is_diverging, all=False
    )
    return synced_is_diverging


def run(exp: Experiment) -> None:

    fabric = ptl.Fabric(
        accelerator=exp.hardware_config.get_accelerator(),
        devices=exp.hardware_config.get_num_devices(),
        num_nodes=1,
        strategy="auto",
    )
    fabric.launch()
    loginfo_on_r0(fabric, f"Using device {fabric.device}")

    data_logger: DataLogger
    if fabric.global_rank == 0:
        data_logger = WandbDataLogger(experiment=exp)
    else:
        data_logger = DummyDataLogger()

    with record_function("initialization"):
        loginfo_on_r0(fabric, "=" * 80)
        loginfo_on_r0(fabric, "Initializing experiment:")
        loginfo_on_r0(fabric, pprint.pformat(exp.loggable_dict(), indent=4))
        loginfo_on_r0(fabric, "=" * 80)
        exp_state = initialize(exp, fabric)

    with record_function("first evaluation"):
        loginfo_on_r0(fabric, "Initial evaluation...")
        eval_and_log(fabric, exp, exp_state, {}, data_logger)

    with record_function("training"):
        loginfo_on_r0(fabric, "Starting training...")
        for t in range(1, exp.steps + 1):
            live_loss, exp_state = training_step(fabric, exp, exp_state)

            if should_early_stop(fabric, live_loss):
                break

            if t % exp.eval_every == 0:
                with record_function("eval"):
                    extra_dict = {"live_loss": live_loss}
                    eval_and_log(fabric, exp, exp_state, extra_dict, data_logger)

    data_logger.finish(exit_code=0)


def initialize(exp: Experiment, fabric: ptl.Fabric) -> ExperimentState:

    loginfo_on_r0(fabric, "Initialization...")
    bs_info = exp.hardware_config.get_batch_size_info(exp.problem)

    seed = exp.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    loginfo_on_r0(fabric, "Loading the dataset...")
    tr_tr_dl = exp.problem.dataset.get_dataloader(tr_va="tr", b=bs_info.mbatchsize_tr)
    tr_va_dl = exp.problem.dataset.get_dataloader(tr_va="tr", b=bs_info.mbatchsize_va)
    va_va_dl = exp.problem.dataset.get_dataloader(tr_va="va", b=bs_info.mbatchsize_va)

    loginfo_on_r0(fabric, "Loading the model...")
    pytorch_model = exp.problem.model.load_model(
        exp.problem.dataset.input_shape(bs_info.mbatchsize_tr),
        exp.problem.dataset.output_shape(bs_info.mbatchsize_tr),
    )

    loginfo_on_r0(fabric, "Loading optimizer...")
    pytorch_opt = exp.optim.load(pytorch_model)
    ptl_model, ptl_opt = fabric.setup(pytorch_model, pytorch_opt)

    return ExperimentState(
        model=ptl_model,
        optimizer=ptl_opt,
        dataloaders=DataLoaders(
            *fabric.setup_dataloaders(tr_tr_dl, tr_va_dl, va_va_dl)
        ),
        batch_size_info=bs_info,
    )


def eval_and_log(
    fabric: ptl.Fabric,
    exp: Experiment,
    exp_data: ExperimentState,
    extra_dict: Dict[str, Any],
    data_logger: DataLogger,
) -> None:

    time_dict = {
        "epoch": exp_data.iteration_counter.epoch,
        "step": exp_data.iteration_counter.step,
        "step_within_epoch": exp_data.iteration_counter.step_within_epoch,
    }

    with record_function("eval(tr)"):
        reduced_metrics_tr = evaluate(
            fabric=fabric,
            loader=exp_data.dataloaders.tr_va,
            metrics=exp.problem.metrics,
            model=exp_data.model,
        )
    with record_function("eval(va)"):
        reduced_metrics_va = evaluate(
            fabric=fabric,
            loader=exp_data.dataloaders.va_va,
            metrics=exp.problem.metrics,
            model=exp_data.model,
        )

    renamed_tr = {
        f"tr_{str(k.__class__.__name__)}": v for k, v in reduced_metrics_tr.items()
    }
    renamed_va = {
        f"va_{str(k.__class__.__name__)}": v for k, v in reduced_metrics_va.items()
    }

    for dict_to_log in [renamed_tr, renamed_va, time_dict, extra_dict]:
        data_logger.log_data(dict_to_log)
    data_logger.commit()


def evaluate(
    fabric: ptl.Fabric,
    loader: DataLoader,
    metrics: Iterable[Metric],
    model: torch.nn.Module,
) -> Dict[Metric, float | list]:

    running_metrics: Dict[Metric, SumAndCounter] = {
        metric: SumAndCounter(torch.tensor(0.0), torch.tensor(0.0))
        for metric in metrics
    }

    with EvalMode(model), torch.no_grad():
        for _, (features, labels) in enumerate(loader):
            y_pred = model(features)

            for metric in metrics:
                loss, weight = metric(y_pred, labels)
                running_metrics[metric] += SumAndCounter(loss.detach(), weight.detach())

        def reduce(x: SumAndCounter):
            num, den = x.reduce(fabric)
            v = (num / den).cpu()
            return v.tolist() if torch.numel(v) > 1 else v.item()

        reduced_results = {k: reduce(v) for k, v in running_metrics.items()}

        return reduced_results


def training_step(
    fabric: ptl.Fabric,
    exp: Experiment,
    exp_state: ExperimentState,
) -> Tuple[float, ExperimentState]:

    with TrainMode(exp_state.model):
        exp_state.optimizer.zero_grad()
        total_loss_and_count = SumAndCounter.zero()

        for t in range(exp_state.batch_size_info.accumulation_steps):
            is_accumulating = t < exp_state.batch_size_info.accumulation_steps - 1
            with fabric.no_backward_sync(exp_state.model, enabled=is_accumulating):

                features, labels = exp_state.get_batch()
                loss, weight = exp.problem.lossfunc(exp_state.model(features), labels)
                total_loss_and_count += SumAndCounter(loss.detach(), weight)
                fabric.backward(loss)

        tot_loss, tot_count = total_loss_and_count.reduce(fabric)

        for p in exp_state.model.parameters():
            if p.grad is not None:
                p.grad /= tot_count

        train_loss = (tot_loss / tot_count).cpu().item()
        exp_state.optimizer.step()

        return train_loss, exp_state
