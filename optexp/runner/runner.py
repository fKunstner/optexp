import pprint
import random
from typing import Dict, Tuple

import lightning as ptl
import numpy as np
import torch
import torch.nn
from torch.profiler import ProfilerActivity, profile, record_function

from optexp.datasets.dataset import Split
from optexp.datastructures import AdditionalInfo
from optexp.experiment import Experiment
from optexp.metrics.metric import Metric
from optexp.runner.debugging import (
    export_memory_snapshot,
    start_record_memory_history,
    stop_record_memory_history,
    trace_handler,
)
from optexp.runner.exp_state import DataLoaders, ExperimentState
from optexp.runner.utils import (
    EvalMode,
    SumAndCounter,
    TrainMode,
    current_time,
    get_losslike_metrics,
    has_losslike_metrics_to_evaluate,
    loginfo_on_r0,
    make_datalogger,
    regularization,
    should_early_stop,
    tqdm,
)

MetricsDict = Dict[str, float | list[float]]


def run_experiment(
    exp: Experiment, run_profiler: bool = False, run_memory_history: bool = False
) -> ExperimentState:
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
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=6, repeat=1),
            record_shapes=True,
            profile_memory=True,
            use_cuda=True,
            with_stack=True,
            on_trace_ready=trace_handler,
        ) as prof:
            exp_state = run(exp)
        prof.export_chrome_trace("trace.json")
        return exp_state

    if run_memory_history:
        start_record_memory_history()
        exp_state = run(exp)
        export_memory_snapshot()
        stop_record_memory_history()
        return exp_state

    return run(exp)


def fabric_initialize(exp):
    fabric = ptl.Fabric(
        accelerator=exp.hardware_config.get_accelerator(),
        devices=exp.hardware_config.get_num_devices(),
        num_nodes=1,
        strategy="auto",
    )
    fabric.launch()
    return fabric


def run(exp: Experiment) -> ExperimentState:

    fabric = fabric_initialize(exp)
    loginfo_on_r0(fabric, f"Using device {fabric.device}")

    data_logger = make_datalogger(exp, fabric)

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

            if exp.eval_every != 0 and t % exp.eval_every == 0:
                with record_function("eval"):
                    data_logger.log(eval_loop(fabric, exp, exp_state))

            data_logger.commit()

            is_stopping = should_early_stop(fabric, live_loss)
            if is_stopping:
                loginfo_on_r0(
                    fabric, f"Stopping early due to divergence. Live loss = {live_loss}"
                )
                break

    data_logger.finish(exit_code=0, stopped_early=is_stopping)

    return exp_state


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

    loaders = {"tr_tr": train_tr_set_dl}

    if has_losslike_metrics_to_evaluate(exp):
        eval_tr_set_dl = exp.problem.dataset.get_dataloader(
            split="tr", b=bs_info.mbatchsize_va, num_workers=bs_info.workers_va
        )
        loaders["eval_tr"] = eval_tr_set_dl

        if exp.problem.dataset.has_validation_set():
            eval_va_set_dl = exp.problem.dataset.get_dataloader(
                split="va", b=bs_info.mbatchsize_va, num_workers=bs_info.workers_va
            )
            loaders["eval_va"] = eval_va_set_dl

        if exp.problem.dataset.has_test_set():
            eval_te_set_dl = exp.problem.dataset.get_dataloader(
                split="te", b=bs_info.mbatchsize_va, num_workers=bs_info.workers_va
            )
            loaders["eval_te"] = eval_te_set_dl

    processed_loaders = {k: fabric.setup_dataloaders(v) for k, v in loaders.items()}

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
        dataloaders=DataLoaders(
            tr_tr=processed_loaders["tr_tr"],
            eval_tr=processed_loaders.get("eval_tr", None),
            eval_va=processed_loaders.get("eval_va", None),
            eval_te=processed_loaders.get("eval_te", None),
        ),
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
        metrics_dicts.update({m.key("tr"): v for m, v in reduced_metrics_tr.items()})

    if exp.problem.dataset.has_validation_set():
        with record_function("eval(va)"):
            reduced_metrics_va = evaluate(
                fabric=fabric, exp=exp, exp_state=exp_state, split="va"
            )
            metrics_dicts.update(
                {m.key("va"): v for m, v in reduced_metrics_va.items()}
            )

    if exp.problem.dataset.has_test_set():
        with record_function("eval(te)"):
            reduced_metrics_te = evaluate(
                fabric=fabric, exp=exp, exp_state=exp_state, split="te"
            )
            metrics_dicts.update(
                {m.key("te"): v for m, v in reduced_metrics_te.items()}
            )

    return metrics_dicts


def evaluate(
    fabric: ptl.Fabric,
    exp: Experiment,
    exp_state: ExperimentState,
    split: Split,
) -> Dict[Metric, float | list]:

    losslike_metrics = get_losslike_metrics(exp)

    if not has_losslike_metrics_to_evaluate(exp):
        return {}

    loader = exp_state.dataloaders.get_val_dataloader(split)
    model = exp_state.model

    running_sum_metrics: Dict[Metric, SumAndCounter] = {
        metric: SumAndCounter(torch.tensor(0.0), torch.tensor(0.0))
        for metric in losslike_metrics
    }

    with EvalMode(model), torch.no_grad():
        for _, batch in tqdm(enumerate(loader), total=len(loader)):
            cached_forward = exp.problem.datapipe.forward(batch, model)
            additional_info = AdditionalInfo(split, exp, exp_state, cached_forward)
            for metric in losslike_metrics:
                #  out of mem here
                loss, weight = exp.problem.datapipe.compute_metric(
                    data=batch,
                    model=model,
                    metric=metric,
                    additional_info=additional_info,
                )
                running_sum = SumAndCounter(loss.detach(), weight.detach())
                running_sum_metrics[metric] += running_sum

        reduced_results = {
            k: v.reduce_and_divide(fabric) for k, v in running_sum_metrics.items()
        }

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
                mb = exp_state.get_microbatch()
                loss, weight = exp.problem.datapipe.compute_metric(
                    data=mb,
                    model=exp_state.model,
                    metric=exp.problem.lossfunc,
                    additional_info=AdditionalInfo("tr", exp, exp_state),
                )
                total_loss_and_count += SumAndCounter(loss.detach(), weight.detach())
                fabric.backward(loss)

        tot_loss, tot_count = total_loss_and_count.reduce(fabric)

        for p in exp_state.model.parameters():
            if p.grad is not None:
                p.grad /= tot_count

        train_loss = (tot_loss / tot_count).cpu().item()
        exp_state.optimizer.step()
        exp_state.step()
        exp_state.optimizer.zero_grad()
        return train_loss, exp_state
