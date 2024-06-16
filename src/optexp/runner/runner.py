import math
import pprint
import random
import textwrap
import time
import traceback
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    TypeVar,
)

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
                manage_exceptions(exp)

        prof.export_chrome_trace("trace.json")
    else:
        manage_exceptions(exp)


def manage_exceptions(
    exp: Experiment,
    data_logger_class: Type[DataLogger] = DataLogger,
):

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


def run(fabric, exp, data_logger):

    def get_dataloaders():
        b = exp.problem.batch_size
        return (
            exp.problem.dataset.input_shape(b),
            exp.problem.dataset.output_shape(b),
            exp.problem.dataset.get_tensor_dataloader(tr_va="tr", b=b),
            exp.problem.dataset.get_tensor_dataloader(tr_va="va", b=b),
        )

    def get_model():
        return exp.problem.model.load_model(input_shape, output_shape)

    def get_loss_metrics_and_optimizer():
        return (
            exp.problem.lossfunc(),
            [metric() for metric in exp.problem.metrics],
            exp.optim.load(model),
        )

    def info_r0(message: str) -> None:
        if fabric.global_rank == 0:
            get_logger().info(message)

    info_r0(
        textwrap.dedent(
            f"""
            {"=" * 80}
            Initializing experiment:
            {pprint.pformat(asdict_with_class(exp),indent=4)}
            {"=" * 80}
            """
        )
    )

    _apply_seed(exp)

    info_r0("Loading dataset...")
    input_shape, output_shape, tr_dl, va_dl = sync_try_except(fabric, get_dataloaders)

    info_r0("Loading model...")
    model = sync_try_except(fabric, get_model)

    info_r0("Loading loss function, metrics, and optimizers...")
    loss_func, metrics, opt = sync_try_except(fabric, get_loss_metrics_and_optimizer)

    info_r0("Fabric setup...")
    model, opt = fabric.setup(model, opt)
    tr_dl, va_dl = fabric.setup_dataloaders(tr_dl, va_dl, move_to_device=True)

    loader = iter(tr_dl)

    def get_batch():
        nonlocal loader
        try:
            features, labels = next(loader)
        except StopIteration:
            loader = iter(tr_dl)
            features, labels = next(loader)
        return features, labels

    info_r0("Initial evaluation...")

    def compute_metrics():
        return (
            evaluate(loader=tr_dl, metrics=metrics, model=model),
            evaluate(loader=va_dl, metrics=metrics, model=model),
        )

    metrics_tr, metrics_va = sync_try_except(fabric, compute_metrics)
    reduced_metrics_tr = reduce_and_make_dictionary(fabric, metrics_tr, "tr")
    reduced_metrics_va = reduce_and_make_dictionary(fabric, metrics_va, "va")

    synchronised_log(
        fabric,
        data_logger,
        reduced_metrics_tr,
        reduced_metrics_va,
        {"step": 0},
    )
    torch.cuda.empty_cache()

    info_r0("Starting training...")

    def compute_loss():
        features, labels = get_batch()
        b = len(labels)
        loss, weight = to_loss_and_weight(loss_func(model(features), labels), b)
        loss_and_count = SumAndCounter(loss.item(), weight)
        if math.isnan(loss) or math.isinf(loss):
            raise DivergingException()
        return loss, loss_and_count

    for t in range(1, exp.steps + 1):
        opt.zero_grad()

        loss_and_count = SumAndCounter(torch.tensor(0.0), torch.tensor(0.0))
        for _ in range(exp.gradient_acc_steps):

            loss, new_loss_and_count = sync_try_except(fabric, compute_loss)
            loss_and_count += new_loss_and_count
            fabric.backward(loss)

        for p in model.parameters():
            if p.grad is not None:
                p.grad /= loss_and_count.denominator

        train_loss = loss_and_count.reduce(fabric).cpu().item()
        opt.step()
        torch.cuda.empty_cache()

        if t % exp.eval_every == 0:
            metrics_tr, metrics_va = sync_try_except(fabric, compute_metrics)
            reduced_metrics_tr = reduce_and_make_dictionary(fabric, metrics_tr, "tr")
            reduced_metrics_va = reduce_and_make_dictionary(fabric, metrics_va, "va")

            synchronised_log(
                fabric,
                data_logger,
                reduced_metrics_tr,
                reduced_metrics_va,
                {"step": t},
                {"live_train_loss": train_loss},
            )
            torch.cuda.empty_cache()


def reduce_and_make_dictionary(
    fabric, metrics_and_counts_raw: Dict[Metric, SumAndCounter], tr_va: TrVa
) -> Dict[str, Iterable | float]:
    def reduce(x):
        v = x.reduce(fabric).cpu()
        return v.tolist() if torch.numel(v) > 1 else v.item()

    values_by_str: Dict[str, Iterable | float] = {
        f"{tr_va}_{str(k.__class__.__name__)}": reduce(v)
        for k, v in metrics_and_counts_raw.items()
    }
    return values_by_str


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


def _apply_seed(self) -> None:
    np.random.seed(self.seed)
    random.seed(self.seed)
    torch.manual_seed(self.seed)
    torch.cuda.manual_seed_all(self.seed)


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


T = TypeVar("T")


class SynchronizedDivergence(Exception):
    pass


class SynchronizedError(Exception):
    pass


def sync_try_except(fabric: ptl.Fabric, func: Callable[[], T]) -> Optional[T]:
    """Raise an exception on all ranks if any rank raises an exception.

    The function `func` must not use multiprocess communication (reduce, gather, ...).
    If it does, the processes might deadlock because the following can happen:
    - Rank 0 raises an exception and tries to broadcast it to all ranks.
    - Rank 1 did not see the exception and tries to gather/reduce some tensor with other ranks.

    This function re-raises the exception on all ranks, so that all ranks can handle it
    """
    w = fabric.world_size
    all_exceptions: Iterable[Optional[Tuple[Exception, str]]] = [None for _ in range(w)]
    output: Optional[T] = None

    if fabric.world_size == 1:
        try:
            output = func()
        except DivergingException as e:
            raise SynchronizedDivergence from e
        except Exception as e:
            raise SynchronizedError from e
        return output

    try:
        output = func()
    except Exception as exception:
        torch.distributed.all_gather_object(
            obj=(exception, get_trace(exception)),
            object_list=all_exceptions,
        )
        raise exception
    else:
        torch.distributed.all_gather_object(
            obj=None,
            object_list=all_exceptions,
        )
    finally:
        valid_exceptions: List[Tuple[int, Exception, str]] = []
        for i, exp_and_trace in enumerate(all_exceptions):
            if exp_and_trace is not None:
                valid_exceptions.append((i, exp_and_trace[0], exp_and_trace[1]))

        if len(valid_exceptions) > 0:
            if fabric.global_rank >= 0:
                rank, exc, trace = valid_exceptions[0]
                exc_class = exc.__class__.__name__
                if isinstance(exc, DivergingException):
                    raise SynchronizedDivergence(
                        f"Detected {exc_class} on rank {rank}.\n\n"
                        f"Traceback for {exc_class}:\n{trace}"
                    ) from exc
                raise SynchronizedError(
                    f"Detected {exc_class} on rank {rank}.\n\n"
                    f"Traceback for {exc_class}:\n{trace}"
                ) from exc
            raise SystemExit()

    return output


def get_trace(ex: BaseException):
    return "".join(traceback.TracebackException.from_exception(ex).format())
