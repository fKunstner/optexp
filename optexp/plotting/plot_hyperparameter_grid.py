import itertools
import os
import warnings
from typing import Dict, List, Literal, Optional, Tuple

import attrs
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame

from optexp.config import Config
from optexp.experiment import Experiment
from optexp.optim import Optimizer
from optexp.plotting.colors import Colors
from optexp.plotting.style import make_axes
from optexp.results.wandb_data_logger import load_wandb_results


def plot_optim_hyperparam_grids(
    exps: List[Experiment],
    folder_name: str,
    hyperparameter: str = "lr",
    up_to_step: Optional[int] = None,
):
    """
    Args:
        exps: List of experiments to use for the plot
        plot_name: Folder under which the plots will be saved
        up_to_step: Number of steps to plot up to. If None, plot all steps.
    """

    for exp in exps:
        if not hasattr(exp.optim, hyperparameter):
            raise ValueError(
                f"Asked to plot hyperparameter grid for parameter {hyperparameter}. "
                f"Optimizer {exp.optim} does not have hyperparameter {hyperparameter}."
            )

    problem = exps[0].problem
    if not all(exp.problem == problem for exp in exps):
        raise ValueError("All experiments must have the same problem.")

    exps_data = load_wandb_results(exps)

    exps_data = truncate_runs(exps_data, up_to_step)

    for metric, tr_va, logx, logy in itertools.product(
        problem.metrics, ["tr", "va"], [True, False], [True, False]
    ):
        metric_key = f"{tr_va}_{metric.__class__.__name__}"

        fig = make_step_size_grid_for_metric(
            exps_data, hyperparameter, metric_key, logx, logy
        )

        save_dir = Config.get_plots_directory() / folder_name
        os.makedirs(save_dir, exist_ok=True)
        filepath = save_dir / (
            "_".join(
                [
                    metric_key,
                    "logx" if logx else "linx",
                    "logy" if logy else "liny",
                ]
            )
            + ".png"
        )
        fig.savefig(filepath)
        print(f"Saving {filepath}")
        plt.close(fig)


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


def truncate_runs(exps_data, up_to_step):
    if up_to_step is not None:
        for exp in exps_data.keys():
            exp_data = exps_data[exp]
            filter_rows = (exp_data["step"] <= up_to_step) | exp_data["step"].isna()
            exps_data[exp] = exp_data[filter_rows]
    return exps_data


def group_experiments_by_optimizers(
    exps: List[Experiment], hyperparameter: str = "lr"
) -> Dict[Optimizer, List[Experiment]]:
    """Group optimizers together if they differ only in the given hyperparameter.

    Args:
        exps: List of experiments to group by optimizer
        hyperparameter: Name of the Optimizer parameter to group together

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
        opt_with_erased_hp = attrs.evolve(exp.optim, **{hyperparameter: None})
        if opt_with_erased_hp in groups:
            groups[opt_with_erased_hp].append(exp)
        else:
            groups[opt_with_erased_hp] = [exp]
    return groups


def sanitize(xs):
    max_val = 10**50
    xs = np.array(xs, dtype="float")
    xs[xs > max_val] = max_val
    return xs


def make_step_size_grid_for_metric(
    exps_data, hyperparameter, metric_key, logx, logy
) -> plt.Figure:

    fig, ax = make_axes(plt, rel_width=1.0, nrows=1, ncols=1)

    data_range = (
        np.min(metric_at_end(exps_data, metric_key)),
        np.max(metric_at_initialization(exps_data, metric_key)),
    )
    optim_groups = group_experiments_by_optimizers(exps_data.keys(), hyperparameter)
    for i, (optim, exps) in enumerate(optim_groups.items()):
        hps, metrics = get_hp_and_metrics_at_end_per_hp(
            exps,
            exps_data,
            hyperparameter,
            metric_key,
        )
        ax.fill_between(
            hps,
            [np.nanmin(metrics[hp]) for hp in hps],
            [np.nanmax(metrics[hp]) for hp in hps],
            color=Colors.Vibrant.get(i),
            alpha=0.2,
        )
        ax.plot(
            hps,
            [np.median(sanitize(metrics[hp])) for hp in hps],
            label=optim.equivalent_definition(),
            color=Colors.Vibrant.get(i),
            marker="o",
        )

    ax.axhline(data_range[1], color="black", linestyle="--", label="mean at init")
    set_limits(ax, x_y="y", limits=data_range, log=logy)

    hp_values = [getattr(exp.optim, hyperparameter) for exp in exps_data.keys()]
    set_limits(ax, x_y="x", limits=(min(hp_values), max(hp_values)), log=logx)

    ax.set_title(f"Grid for {metric_key}")
    ax.set_xlabel(hyperparameter)
    ax.set_ylabel(metric_key)
    ax.legend()

    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")

    return fig


def get_hp_and_metrics_at_end_per_hp(
    exps: List[Experiment],
    exps_data: Dict[Experiment, DataFrame],
    hyperparameter: str,
    metric_key: str,
) -> Tuple[List[float], Dict[float, List[float]]]:
    hp_values = sorted(list(set(getattr(exp.optim, hyperparameter) for exp in exps)))
    metrics_at_end_for_hp: Dict[float, List[float]] = {hp: [] for hp in hp_values}
    for exp in exps:
        hp = getattr(exp.optim, hyperparameter)
        metrics_at_end_for_hp[hp].append(exps_data[exp][metric_key].iloc[-1])

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
        delta = np.log10(limits[1]) - np.log10(limits[0])
        mult_factor = 10 ** (delta / (100 * (factor - 1)))
        ax_set_limits([limits[0] / mult_factor, limits[1] * mult_factor])
    else:
        delta = limits[1] - limits[0]
        add_factor = delta / (100 * (factor - 1))
        ax_set_limits([limits[0] - add_factor, limits[1] + add_factor])
