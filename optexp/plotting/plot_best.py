import itertools
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame

from optexp.config import Config
from optexp.experiment import Experiment
from optexp.metrics import Metric
from optexp.optim import Optimizer
from optexp.plotting.colors import Colors
from optexp.plotting.style import make_axes
from optexp.plotting.utils import (
    ensure_all_exps_have_hyperparameter,
    ensure_all_exps_have_same_problem,
    flatten,
    get_best_exps_per_group,
    hack_steps_for_logscale,
    save_and_close,
    scale_str,
    set_limits,
    set_scale,
    set_ylimits_to_fit_data_range,
    truncate_runs,
)
from optexp.results.wandb_data_logger import load_wandb_results


def plot_metrics_over_time_for_best(
    exps: List[Experiment],
    folder_name: str,
    hp: str = "lr",
    regularized: bool = False,
    step: Optional[int] = None,
    metric_key: Optional[str] = None,
):
    """
    Args:
        exps: List of experiments to use for the plot.
        folder_name: Folder under which the plots will be saved.
        hp: Optimizer hyperparameter to optimize over. Defaults to "lr"
        regularized: Whether to consider the regularization penalty in the loss
        step: Number of steps to plot up to. If None, plot all steps.
        metric_key: Metric to use to decide which hyperparameter is best,
            given in the format "[tr|va]_LossName". Defaults to the training loss.
    """

    ensure_all_exps_have_hyperparameter(exps, hp)
    problem = ensure_all_exps_have_same_problem(exps)

    exps_data = load_wandb_results(exps)
    exps_data = truncate_runs(exps_data, step)

    best_exps_per_group = get_best_exps_per_group(
        exps_data, hp, problem, regularized, metric_key
    )

    steps_subfolder = "max_steps" if step is None else f"{step}_steps"
    folder = Config.get_plots_directory() / folder_name / "best" / steps_subfolder
    os.makedirs(folder, exist_ok=True)

    for metric, tr_va, log_x_y in itertools.product(
        problem.metrics, ["tr", "va"], itertools.product([True, False], [True, False])
    ):
        key = metric.key(tr_va)
        if metric.is_scalar():
            fig = make_best_plot_for_metric(
                best_exps_per_group, exps_data, metric, key, log_x_y
            )
        else:
            fig = make_best_plot_for_non_scalar_metric(
                best_exps_per_group, exps_data, metric, key, log_x_y
            )
        save_and_close(
            fig, folder, [key, f"{scale_str(log_x_y[0])}x", f"{scale_str(log_x_y[1])}y"]
        )


def make_best_plot_for_non_scalar_metric(
    best_exps_per_group: Dict[Optimizer, List[Experiment]],
    exps_data: Dict[Experiment, DataFrame],
    metric: Metric,
    metric_key: str,
    log_x_y: Tuple[bool, bool] = (False, False),
) -> plt.Figure:

    n_groups = len(best_exps_per_group)
    fig, axes = make_axes(plt, rel_width=1.0, nrows=1, ncols=n_groups)
    if n_groups == 1:
        axes = [[axes]]

    for i, (opt, exps) in enumerate(best_exps_per_group.items()):
        reduced_dfs = [exps_data[exp][["step", metric_key]].dropna() for exp in exps]
        steps = np.array(list(reduced_dfs[0]["step"]), dtype=float)
        steps = hack_steps_for_logscale(steps)
        values = np.stack([np.stack(df[metric_key].to_numpy()) for df in reduced_dfs])

        # values : [n_exps, n_steps, n_series]
        n_series = values.shape[2]

        for j in range(n_series):
            axes[0][i].fill_between(
                steps,
                np.min(values[:, :, j], axis=0),
                np.max(values[:, :, j], axis=0),
                color=Colors.viridis(j, n_series),
                alpha=0.2,
            )
            axes[0][i].plot(
                steps,
                np.median(values[:, :, j], axis=0),
                color=Colors.viridis(j, n_series),
                label=exps[0].optim.equivalent_definition(),
            )
        axes[0][i].set_title(opt.equivalent_definition())

    reduced_exp_data = {
        exp: data
        for exp, data in exps_data.items()
        if exp in flatten(best_exps_per_group.values())
    }

    min_and_max_step_values = (
        min(data["step"].min() for exp, data in exps_data.items()),
        max(data["step"].max() for exp, data in exps_data.items()),
    )
    for ax in flatten(axes):
        set_ylimits_to_fit_data_range(
            ax, reduced_exp_data, metric, metric_key, log_x_y[1]
        )
        set_limits(
            ax,
            x_y="x",
            limits=min_and_max_step_values,
            log=log_x_y[0],
            factor=1.0,
        )
        set_scale(ax, log_x_y)
        ax.set_xlabel("Steps")
        ax.set_ylabel(metric_key)

    return fig


def make_best_plot_for_metric(
    best_exps_per_group: Dict[Optimizer, List[Experiment]],
    exps_data: Dict[Experiment, DataFrame],
    metric: Metric,
    metric_key: str,
    log_x_y: Tuple[bool, bool] = (False, False),
) -> plt.Figure:
    fig, ax = make_axes(plt, rel_width=1.0, nrows=1, ncols=1)

    for i, (_, exps) in enumerate(best_exps_per_group.items()):
        reduced_dfs = [exps_data[exp][["step", metric_key]].dropna() for exp in exps]
        steps = np.array(list(reduced_dfs[0]["step"]), dtype=float)
        steps = hack_steps_for_logscale(steps)
        values = np.stack([df[metric_key] for df in reduced_dfs])

        ax.fill_between(
            steps,
            np.min(values, axis=0),
            np.max(values, axis=0),
            color=Colors.Vibrant.get(i),
            alpha=0.2,
        )
        ax.plot(
            steps,
            np.median(values, axis=0),
            color=Colors.Vibrant.get(i),
            label=exps[0].optim.equivalent_definition(),
        )

    reduced_exp_data = {
        exp: data
        for exp, data in exps_data.items()
        if exp in flatten(best_exps_per_group.values())
    }

    set_ylimits_to_fit_data_range(ax, reduced_exp_data, metric, metric_key, log_x_y[1])

    min_and_max_step_values = (
        min(data["step"].min() for exp, data in exps_data.items()),
        max(data["step"].max() for exp, data in exps_data.items()),
    )
    set_limits(
        ax,
        x_y="x",
        limits=min_and_max_step_values,
        log=log_x_y[0],
        factor=1.0,
    )
    set_scale(ax, log_x_y)

    ax.set_title(f"Grid for {metric_key}")
    ax.set_xlabel("Steps")
    ax.set_ylabel(metric_key)
    ax.legend()

    return fig
