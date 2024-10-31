import math
import pprint
import random
from typing import Dict, Tuple

import lightning as ptl
import numpy as np
import torch
import torch.nn
from torch.profiler import ProfilerActivity, profile, record_function

from optexp.datasets.dataset import Split
from optexp.experiment import Experiment
from optexp.metrics.metric import Metric
from optexp.optim.optimizer import Regularizable
from optexp.results.data_logger import DataLogger, DummyDataLogger
from optexp.results.main_data_logger import MainDataLogger
from optexp.runner.exp_state import DataLoaders, ExperimentState
from optexp.runner.utils import EvalMode, SumAndCounter, TrainMode, loginfo_on_r0

MetricsDict = Dict[str, float | list[float]]


def run_experiment(exp: Experiment, run_profiler: bool = False) -> ExperimentState:
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
            exp_state = run(exp)
        prof.export_chrome_trace("trace.json")
        return exp_state
    return run(exp)


def should_early_stop(fabric: ptl.Fabric, live_loss: float):
    is_diverging = math.isnan(live_loss) or math.isinf(live_loss)
    synced_is_diverging = fabric.strategy.reduce_boolean_decision(
        is_diverging, all=False
    )
    return synced_is_diverging


def run(exp: Experiment) -> ExperimentState:

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
        data_logger = MainDataLogger(experiment=exp)
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
        data_logger.log(current_time(exp_state))
        data_logger.log(regularization(exp, exp_state))
        data_logger.log(eval_loop(fabric, exp, exp_state))
        data_logger.commit()

    is_stopping: bool = False
    with record_function("training"):
        loginfo_on_r0(fabric, "Starting training...")
        for t in range(1, exp.steps + 1):
            live_loss, exp_state = training_step(fabric, exp, exp_state)

            data_logger.log({"live_loss": live_loss})
            data_logger.log(current_time(exp_state))
            data_logger.log(regularization(exp, exp_state))

            if t % exp.eval_every == 0:
                with record_function("eval"):
                    data_logger.log(eval_loop(fabric, exp, exp_state))

            data_logger.commit()

            is_stopping = should_early_stop(fabric, live_loss)
            if is_stopping:
                break

    data_logger.finish(exit_code=0, stopped_early=is_stopping)

    return exp_state


def current_time(exp_state):
    return {
        "epoch": exp_state.iteration_counter.epoch,
        "step": exp_state.iteration_counter.step,
        "step_within_epoch": exp_state.iteration_counter.step_within_epoch,
    }


def regularization(exp, exp_state):
    if isinstance(exp.optim, Regularizable):
        return {
            "regularization": exp.optim.regularizer_loss(exp_state.model).cpu().item()
        }
    return {}


def initialize(exp: Experiment, fabric: ptl.Fabric) -> ExperimentState:

    loginfo_on_r0(fabric, "Initialization...")
    bs_info = exp.hardware_config.get_batch_size_info(exp.problem)

    seed = exp.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    loginfo_on_r0(fabric, "Loading the dataset...")
    if hasattr(exp.problem.dataset, "get_truncation_information"):
        truncation_info = exp.problem.dataset.get_truncation_information()
        loginfo_on_r0(fabric, f"  Dataset might be truncated: {truncation_info}")

    train_tr_set_dl = exp.problem.dataset.get_dataloader(
        split="tr", b=bs_info.mbatchsize_tr, num_workers=bs_info.workers_tr
    )
    eval_tr_set_dl = exp.problem.dataset.get_dataloader(
        split="tr", b=bs_info.mbatchsize_va, num_workers=bs_info.workers_va
    )
    eval_va_set_dl = exp.problem.dataset.get_dataloader(
        split="va", b=bs_info.mbatchsize_va, num_workers=bs_info.workers_va
    )

    loaders = [train_tr_set_dl, eval_tr_set_dl, eval_va_set_dl]

    if exp.problem.dataset.has_test_set():
        eval_te_set_dl = exp.problem.dataset.get_dataloader(
            split="te", b=bs_info.mbatchsize_va, num_workers=bs_info.workers_va
        )
        loaders.append(eval_te_set_dl)

    loginfo_on_r0(fabric, "Loading the model...")
    pytorch_model = exp.problem.model.load_model(
        exp.problem.dataset.data_input_shape(bs_info.mbatchsize_tr),
        exp.problem.dataset.model_output_shape(bs_info.mbatchsize_tr),
    )

    loginfo_on_r0(fabric, "Loading optimizer...")
    pytorch_opt = exp.optim.load(pytorch_model)
    ptl_model, ptl_opt = fabric.setup(pytorch_model, pytorch_opt)

    exp_state = ExperimentState(
        model=ptl_model,
        optimizer=ptl_opt,
        dataloaders=DataLoaders(*fabric.setup_dataloaders(*loaders)),
        batch_size_info=bs_info,
    )

    if exp.problem.init_callback is not None:

        def log(message):
            loginfo_on_r0(fabric, message)

        exp_state = exp.problem.init_callback(exp, exp_state, log)
    return exp_state


def eval_loop(
    fabric: ptl.Fabric,
    exp: Experiment,
    exp_state: ExperimentState,
) -> MetricsDict:

    metrics_dicts: MetricsDict = {}
    with record_function("eval(tr)"):
        reduced_metrics_tr = evaluate(
            fabric=fabric, exp=exp, exp_state=exp_state, split="tr"
        )
        metrics_dicts.update(
            {
                f"tr_{str(k.__class__.__name__)}": v
                for k, v in reduced_metrics_tr.items()
            }
        )

    with record_function("eval(va)"):
        reduced_metrics_va = evaluate(
            fabric=fabric, exp=exp, exp_state=exp_state, split="va"
        )
        metrics_dicts.update(
            {
                f"va_{str(k.__class__.__name__)}": v
                for k, v in reduced_metrics_va.items()
            }
        )

    if exp.problem.dataset.has_test_set():
        with record_function("eval(te)"):
            reduced_metrics_te = evaluate(
                fabric=fabric, exp=exp, exp_state=exp_state, split="te"
            )
            metrics_dicts.update(
                {
                    f"te_{str(k.__class__.__name__)}": v
                    for k, v in reduced_metrics_te.items()
                }
            )

    return metrics_dicts


def evaluate(
    fabric: ptl.Fabric,
    exp: Experiment,
    exp_state: ExperimentState,
    split: Split,
) -> Dict[Metric, float | list]:

    loader = exp_state.dataloaders.get_val_dataloader(split)
    metrics = exp.problem.metrics
    model = exp_state.model

    running_metrics: Dict[Metric, SumAndCounter] = {
        metric: SumAndCounter(torch.tensor(0.0), torch.tensor(0.0))
        for metric in metrics
    }

    with EvalMode(model), torch.no_grad():
        for _, batch in enumerate(loader):

            cached_forward = exp.problem.datapipe.forward(batch, model, split=split)
            for metric in metrics:
                loss, weight = exp.problem.datapipe.compute_metric(
                    data=batch,
                    model=model,
                    metric=metric,
                    split=split,
                    cached_forward=cached_forward,
                )
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
                loss, weight = exp.problem.datapipe.compute_loss(
                    data=exp_state.get_batch(),
                    model=exp_state.model,
                    lossfunc=exp.problem.lossfunc,
                    split="tr",
                )
                total_loss_and_count += SumAndCounter(loss.detach(), weight)
                fabric.backward(loss)

        tot_loss, tot_count = total_loss_and_count.reduce(fabric)

        for p in exp_state.model.parameters():
            if p.grad is not None:
                p.grad /= tot_count

        train_loss = (tot_loss / tot_count).cpu().item()
        exp_state.optimizer.step()

        return train_loss, exp_state
