import warnings
from typing import Dict, Iterable, List, Literal, Optional, Tuple

import attrs
import numpy as np
from matplotlib import pyplot as plt
from pandas import DataFrame

from optexp.experiment import Experiment
from optexp.metrics import Metric
from optexp.optim import Optimizer
from optexp.problem import Problem


def infer_metric_key(
    split, metric: Metric, exp_data: Dict[Experiment, DataFrame]
) -> str:
    """For backwards compatibillity.

    Metric keys where initially "{split}_{metric.__class__.__name__}",
    but are now "{split}_{str(metric)}" to allow for metrics with arguments.

    This function infers the metric key from the experiment keys.
    """
    old_key = f"{split}_{metric.__class__.__name__}"
    new_key = f"{split}_{str(metric)}"

    first_df = next(iter(exp_data.values()))

    if old_key in first_df.keys():
        return old_key
    if new_key in first_df.keys():
        return new_key
    raise ValueError(f"Could not find metric key for {metric} in {exp_data.keys()}")


def ensure_all_exps_have_same_problem(exps):
    problem = exps[0].problem
    if not all(exp.problem == problem for exp in exps):
        raise ValueError("All experiments must have the same problem.")
    return problem


def ensure_all_exps_have_hyperparameter(exps, hp):
    for exp in exps:
        if not hasattr(exp.optim, hp):
            raise ValueError(
                f"Asked to plot hyperparameter grid for parameter {hp}. "
                f"Optimizer {exp.optim} does not have hyperparameter {hp}."
            )


def check_seeds(experiments):
    unique_optimizers = list(set(exp.optim for exp in experiments))
    seeds_per_optimizers = {}
    for opt in unique_optimizers:
        seeds_per_optimizers[opt] = [
            exp.seed for exp in experiments if exp.optim == opt
        ]

    seeds_for_opt_0 = seeds_per_optimizers[unique_optimizers[0]]
    for opt, seeds in seeds_per_optimizers.items():
        if len(seeds) != len(seeds_for_opt_0):
            warnings.warn(
                "The number of seeds for each optimizer is inconsistent. "
                "The hyperparameter grid plot might be misleading. "
                f"Optimizer {unique_optimizers[0]} has seeds {seeds_for_opt_0} "
                f"while optimizer {opt} has seeds {seeds}."
            )


def group_experiments_by_optimizers(
    exps: Iterable[Experiment], hp: str = "lr"
) -> Dict[Optimizer, List[Experiment]]:
    """Group optimizers together if they differ only in the given hyperparameter.

    Args:
        exps: List of experiments to group by optimizer
        hp: Name of the Optimizer hyperparameter to group together

    Returns:
        Dictionary mapping experiment with the hyperparameter replaced by None
        to a list of optimizers with the same settings, except in the given hyperparameter.
        For example:

            groups = get_optimizers_groups([
                SGD(lr=0.1, momentum=0),
                SGD(lr=1.0, momentum=0),
                SGD(lr=0.1, momentum=0.9),
                SGD(lr=1.0, momentum=0.9),
            ], hyperparameter="lr")
            groups == {
                SGD(lr=None, momentum=0): [
                    SGD(lr=0.1, momentum=0),
                    SGD(lr=1.0, momentum=0)
                ],
                SGD(lr=None, momentum=0.9): [
                    SGD(lr=0.1, momentum=0.9),
                    SGD(lr=1.0, momentum=0.9)
                ],
            }
    """
    groups: Dict[Optimizer, List[Experiment]] = {}
    for exp in exps:
        opt_with_erased_hp = attrs.evolve(exp.optim, **{hp: None})
        if opt_with_erased_hp in groups:
            groups[opt_with_erased_hp].append(exp)
        else:
            groups[opt_with_erased_hp] = [exp]
    return groups


def sanitize(xs, metric: Metric):
    xs = np.array(xs, dtype="float")

    if metric.smaller_is_better():
        max_val = 10**10
        xs[np.isnan(xs)] = max_val
        xs[xs > max_val] = max_val
    else:
        min_val = 0
        xs[np.isnan(xs)] = min_val
        xs[xs < min_val] = min_val
    return xs


def get_hp_and_metrics_at_end_per_hp(
    exps_data: Dict[Experiment, DataFrame], hp: str, regularized: bool, metric_key: str
) -> Tuple[List[float], Dict[float, List[float]]]:
    hp_values = sorted(list(set(getattr(exp.optim, hp) for exp in exps_data.keys())))
    metrics_at_end_for_hp: Dict[float, List[float]] = {hp: [] for hp in hp_values}
    for exp, exp_data in exps_data.items():
        val = getattr(exp.optim, hp)
        data = exp_data[metric_key].iloc[-1]
        if regularized:
            data += 0.5 * exp_data["regularization"].iloc[-1]
        metrics_at_end_for_hp[val].append(data)
    return hp_values, metrics_at_end_for_hp


def metric_at_initialization(exps_data, metric_key):
    return [exp_data[metric_key].iloc[0] for exp_data in exps_data.values()]


def metric_at_end(exps_data, metric_key):
    return [exp_data[metric_key].iloc[-1] for exp_data in exps_data.values()]


def set_limits(
    ax,
    x_y: Literal["x", "y"],
    limits: Tuple[float, float],
    log: bool,
    factor: float = 1.1,
):
    if x_y not in ["x", "y"]:
        raise ValueError(f"Invalid value for x_y. Expected 'x' or 'y', got: {x_y}.")
    ax_set_limits = ax.set_xlim if x_y == "x" else ax.set_ylim

    if log:
        if limits[0] == 0:
            warnings.warn(
                "Attempting to set limits to log scale with 0 in the range. "
                "Aborting, leaving limits unchanged."
            )
            return

        delta = np.log10(limits[1]) - np.log10(limits[0])
        if np.isclose(factor, 1.0):
            mult_factor = 1
        else:
            mult_factor = 10 ** (delta / (100 * (factor - 1)))

        ax_set_limits([limits[0] / mult_factor, limits[1] * mult_factor])
    else:
        delta = limits[1] - limits[0]
        add_factor = delta * (factor - 1)
        ax_set_limits([limits[0] - add_factor, limits[1] + add_factor])


def truncate_runs(exps_data, up_to_step):
    if up_to_step is not None:
        for exp in exps_data.keys():
            exp_data = exps_data[exp]
            filter_rows = (exp_data["step"] <= up_to_step) | exp_data["step"].isna()
            exps_data[exp] = exp_data[filter_rows]
    return exps_data


def save_and_close(fig, save_dir, filename_elements):
    for file_type in ["pdf"]:
        filepath = save_dir / ("_".join(filename_elements) + "." + file_type)
        fig.savefig(filepath)
        print(f"Saving {filepath}")
    plt.close(fig)


def scale_str(log: bool):
    return "log" if log else "lin"


def validate_metric_key(metric_key, problem) -> Tuple[str, Metric]:
    if metric_key is None:
        if problem.lossfunc not in problem.metrics:
            raise ValueError(
                f"The loss function used for the problem was not logged "
                f"(it does not appear in the metrics) for problem {problem}."
            )
        return f"tr_{problem.lossfunc.__class__.__name__}", problem.lossfunc

    metrics_elements = metric_key.split("_")
    format_is_valid = all(
        [
            len(metrics_elements) == 2,
            metrics_elements[0] in ["tr", "va"],
        ]
    )
    if not format_is_valid:
        raise ValueError(
            f"Invalid format for the metric. Got {metric_key}, expected '[tr|va]_{{loss}}'."
        )
    available_metrics = {
        metric.__class__.__name__: metric for metric in problem.metrics
    }
    metric = None
    for available_metric, metric_class in available_metrics.items():
        if available_metric in metric_key:
            metric = metric_class

    if metric is None:
        raise ValueError(
            f"Metric {metrics_elements[1]} not found in the problem metrics. "
            f"Available: {available_metrics}."
        )
    return metric_key, metric


def get_best_exps_per_group(
    exps_data: Dict[Experiment, DataFrame],
    hp: str,
    problem: Problem,
    regularized: bool,
    metric_key: Optional[str] = None,
) -> Dict[Optimizer, List[Experiment]]:
    metric_key, best_metric = validate_metric_key(metric_key, problem)

    optim_groups = group_experiments_by_optimizers(exps_data.keys(), hp)

    best_hp_per_group: Dict[Optimizer, float] = {}
    for opt, group in optim_groups.items():
        group_exps_data = {exp: exps_data[exp] for exp in group}
        best_hp_per_group[opt] = get_best_hp(
            group_exps_data,
            hp,
            regularized,
            metric_key,
            best_metric.smaller_is_better(),
        )

    def filter_hp(exps, value):
        return [exp for exp in exps if getattr(exp.optim, hp) == value]

    best_exps_per_group = {
        optim: filter_hp(group, best_hp_per_group[optim])
        for optim, group in optim_groups.items()
    }

    return best_exps_per_group


def get_best_hp(
    group_exps_data: Dict[Experiment, DataFrame],
    hp: str,
    regularized: bool,
    metric_key: str,
    smaller_better: bool,
) -> float:
    _, metrics = get_hp_and_metrics_at_end_per_hp(
        group_exps_data, hp, regularized, metric_key
    )

    def worst_case(values_at_end: List[float]):
        for i, v in enumerate(values_at_end):
            if np.isnan(v):
                if smaller_better:
                    values_at_end[i] = np.inf
                else:
                    values_at_end[i] = -1 * np.inf
        if smaller_better:
            return max(values_at_end)
        return min(values_at_end)

    def is_better(new, old):
        if smaller_better:
            return new < old
        return new > old

    best_hp: float = np.nan
    best_value: float = np.inf if smaller_better else -np.inf

    for hp_val, metric_values in metrics.items():
        value_for_hp = worst_case(metric_values)
        if is_better(value_for_hp, best_value):
            best_hp, best_value = hp_val, value_for_hp

    return best_hp


def set_ylimits_to_fit_data_range(ax, exps_data, metric, metric_key, log_y):

    init_values = metric_at_initialization(exps_data, metric_key)
    end_values = metric_at_end(exps_data, metric_key)
    if metric.smaller_is_better():
        low_vals = end_values
        high_vals = init_values
    else:
        low_vals = init_values
        high_vals = end_values

    if isinstance(low_vals, list):
        low_vals = np.stack(low_vals).flatten()
    if isinstance(high_vals, list):
        high_vals = np.stack(high_vals).flatten()

    data_range = (
        np.nanmin(low_vals),
        np.nanmax(high_vals),
    )
    if log_y and data_range[0] <= 0:
        data_range = (
            np.nanmin([val for val in low_vals if val >= 0]),
            data_range[1],
        )

    ax.axhline(
        np.median(init_values), color="black", linestyle="--", label="median init"
    )
    set_limits(ax, x_y="y", limits=data_range, log=log_y)


def set_scale(ax, log_x_y):
    if log_x_y[0]:
        ax.set_xscale("log")
    if log_x_y[1]:
        ax.set_yscale("log")


def hack_steps_for_logscale(steps):
    """
    Replaces values smaller than 1 with 0.1 so that they can be seen in a log-x plot
    """
    steps[steps < 1] = 0.1
    return steps


def flatten(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]
