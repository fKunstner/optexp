import math
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple

import lightning as ptl
import numpy as np
import torch
import torch.nn
from torch import Tensor
from torch.profiler import ProfilerActivity, profile, record_function

from optexp.config import get_device, get_logger
from optexp.datasets.dataset import TrVa
from optexp.experiments.experiment import Experiment
from optexp.experiments.fabric_helpers import gather, gather_bool, reduce_tensor
from optexp.loggers import DataLogger
from optexp.loggers.asdict_with_classes import asdict_with_class
from optexp.problems import DivergingException
from optexp.problems.metrics import Metric


@dataclass
class SumAndCounter:
    numerator: Tensor
    denominator: Tensor

    def __add__(self, other: "SumAndCounter") -> "SumAndCounter":
        return SumAndCounter(
            self.numerator + other.numerator, self.denominator + other.denominator
        )

    def reduce(self, fabric: ptl.Fabric, reduce_op="sum") -> Tensor:
        numerator = reduce_tensor(fabric, self.numerator, reduce_op=reduce_op)
        denominator = reduce_tensor(fabric, self.denominator, reduce_op=reduce_op)
        return numerator / denominator

    def result(self) -> Tensor:
        return self.numerator / self.denominator

    @staticmethod
    def from_output(output: Tensor | Tuple[Tensor, Tensor]):
        if isinstance(output, tuple) and len(output) == 2:
            return SumAndCounter(output[0], output[1])
        return SumAndCounter(output, torch.tensor(1))


def _apply_seed(self) -> None:
    """Apply the seed to all random number generators.

    To be called before the experiments is run.
    """
    np.random.seed(self.seed)
    random.seed(self.seed)
    torch.manual_seed(self.seed)
    torch.cuda.manual_seed_all(self.seed)


def tensor_any(param: Tensor):
    return any(param) if len(param.shape) > 0 else bool(param)


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


def run_experiment(exp: Experiment, run_profiler: bool = False) -> None:
    """Performs a run of the experiments. Generates the run-id, applies the seed and
    creates the data logger. Initializes the problem and optimizer and optimizes the
    problem given the optimizer for the defined amount of epochs. Logs the loss function
    values/metrics returned during the eval and training. Catches any exception raised
    during this process and logs it before exiting.

    Raises:
        BaseException: Raised when user Ctrl+C when experiments is running.
    """
    run_id = time.strftime("%Y-%m-%d--%H-%M-%S")

    if run_profiler:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            with record_function("main_run"):
                _run_experiment(exp, run_id)

        prof.export_chrome_trace("trace.json")
    else:
        _run_experiment(exp, run_id)


def _run_experiment(exp, run_id):
    _apply_seed(exp)
    fabric = ptl.Fabric(
        accelerator=get_device(),
        devices=exp.devices,
        num_nodes=exp.nodes,
        strategy=exp.strategy,
    )
    fabric.launch()
    exceptions = {
        "DivergenceException": False,
        "Exception": False,
        "BaseException": False,
    }
    experiment_success = True
    # only rank 0 gets a real one
    data_logger = None
    fabric.barrier()
    if fabric.global_rank == 0:
        data_logger = DataLogger(
            config_dict=asdict_with_class(exp),
            group=exp.group,
            run_id=run_id,
            exp_id=exp.exp_id(),
            save_directory=exp.save_directory(),
            wandb_autosync=exp.wandb_autosync,
        )

        get_logger().info("=" * 80)
        get_logger().info(f"Initializing  experiments: {exp}")
        get_logger().info("=" * 80)
    fabric.barrier()
    try:
        with fabric.rank_zero_first():
            b = exp.problem.batch_size
            get_logger().info("Loading dataset...")
            tr_loader = exp.problem.dataset.get_dataloader(tr_va="tr", b=b)
            va_loader = exp.problem.dataset.get_dataloader(tr_va="va", b=b)
            input_shape = exp.problem.dataset.input_shape(b)
            output_shape = exp.problem.dataset.output_shape(b)
            get_logger().info("Loading model...")
            model = exp.problem.model.load_model(input_shape, output_shape)
            loss_func = exp.problem.lossfunc()
            metrics = [metric() for metric in exp.problem.metrics]

        opt = exp.optim.load(model)

        get_logger().info("Fabric setup...")
        model, opt = fabric.setup(model, opt)
        tr_loader, va_loader = fabric.setup_dataloaders(
            tr_loader, va_loader, move_to_device=True
        )

        get_logger().info("Initial evaluation...")
        synchronised_log(
            fabric,
            data_logger,
            compute_metrics(fabric, metrics, model, tr_loader, tr_va="tr"),
            compute_metrics(fabric, metrics, model, va_loader, tr_va="va"),
            {"step": 0},
        )
        torch.cuda.empty_cache()

        loader = iter(tr_loader)

        def get_batch():
            nonlocal loader
            try:
                features, labels = next(loader)
            except StopIteration:
                loader = iter(tr_loader)
                features, labels = next(loader)
            return features, labels

        opt.zero_grad()

        get_logger().info("Starting training...")
        for t in range(1, exp.steps + 1):
            opt.zero_grad()
            train_loss = compute_minibatch_loss_and_gradient(
                exp.gradient_acc_steps,
                fabric,
                get_batch,
                loss_func,
                model,
            )
            opt.step()
            torch.cuda.empty_cache()

            if t % exp.eval_every == 0:
                synchronised_log(
                    fabric,
                    data_logger,
                    compute_metrics(fabric, metrics, model, tr_loader, tr_va="tr"),
                    compute_metrics(fabric, metrics, model, va_loader, tr_va="va"),
                    {"step": t},
                    {"live_train_loss": train_loss},
                )
                torch.cuda.empty_cache()

            check_exceptions(fabric, exceptions)

    except DivergingException as e:
        exceptions["DivergenceException"] = True
        experiment_success = False
        if fabric.global_rank == 0:
            get_logger().warning("TERMINATING EARLY. Diverging.")
            get_logger().warning(e, exc_info=True)
            if data_logger is not None:
                data_logger.save(exit_code=0)
    except Exception as e:
        exceptions["Exception"] = True
        experiment_success = False
        if fabric.global_rank == 0:
            get_logger().error("TERMINATING. Encountered error")
            get_logger().error(e, exc_info=True)
            if data_logger is not None:
                data_logger.save(exit_code=1)
    except BaseException as e:
        exceptions["BaseException"] = True
        experiment_success = False
        if fabric.global_rank == 0:
            get_logger().error("TERMINATING. System exit")
            get_logger().error(e, exc_info=True)
            if data_logger is not None:
                data_logger.save(exit_code=1)
    finally:
        fabric.barrier()
        if fabric.global_rank == 0:
            get_logger().info("All Processes Terminating")
    experiment_success = tensor_any(gather(fabric, experiment_success))
    if fabric.global_rank == 0 and experiment_success:
        get_logger().info("Experiment finished.")
        if data_logger is not None:
            data_logger.save(exit_code=0)


def compute_metrics(fabric, metrics, model, loader, tr_va: TrVa):
    metrics_and_counts_eval_train = evaluate(
        loader=loader,
        metrics=metrics,
        model=model,
    )
    values_by_metrics: Dict[Metric, Tensor] = {
        k: v.reduce(fabric).cpu() for k, v in metrics_and_counts_eval_train.items()
    }
    values_by_str: Dict[str, Iterable | float] = {
        f"{tr_va}_{str(k.__class__.__name__)}": (
            v.tolist() if torch.numel(v) > 1 else v.item()
        )
        for k, v in values_by_metrics.items()
    }
    return values_by_str


def compute_minibatch_loss_and_gradient(
    gradient_acc_steps,
    fabric,
    get_batch,
    loss_func,
    model,
):
    loss_and_count = SumAndCounter(torch.tensor(0.0), torch.tensor(0.0))

    for _ in range(gradient_acc_steps):
        features, labels = get_batch()
        b = len(labels)
        loss, weight = to_loss_and_weight(loss_func(model(features), labels), b)
        loss_and_count += SumAndCounter(loss.item(), weight)

        if math.isnan(loss) or math.isinf(loss):
            raise DivergingException()

        fabric.backward(loss)

    for p in model.parameters():
        if p.grad is not None:
            p.grad /= loss_and_count.denominator

    return loss_and_count.reduce(fabric).cpu().item()


def to_loss_and_weight(output: Tensor | Tuple[Tensor, Tensor], default_weight: float):
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


def check_exceptions(fabric, exceptions: Dict[str, bool]) -> None:
    all_exceptions: Dict[str, Tensor] = gather_bool(fabric, exceptions)
    if tensor_any(all_exceptions["DivergenceException"]):
        raise DivergingException()
    if tensor_any(all_exceptions["Exception"]):
        raise Exception()
    if tensor_any(all_exceptions["BaseException"]):
        raise BaseException


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
