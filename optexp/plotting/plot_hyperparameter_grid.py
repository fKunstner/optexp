import itertools
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame

from optexp.config import Config
from optexp.experiment import Experiment
from optexp.plotting.colors import Colors
from optexp.plotting.style import make_axes
from optexp.plotting.utils import (
    ensure_all_exps_have_hyperparameter,
    ensure_all_exps_have_same_problem,
    get_hp_and_metrics_at_end_per_hp,
    group_experiments_by_optimizers,
    sanitize,
    save_and_close,
    scale_str,
    set_limits,
    set_scale,
    set_ylimits_to_fit_data_range,
    truncate_runs,
)
from optexp.results.wandb_data_logger import load_wandb_results


def plot_optim_hyperparam_grids(
    exps: List[Experiment],
    folder_name: str,
    hp: str = "lr",
    step: Optional[int] = None,
):
    """
    Args:
        exps: List of experiments to use for the plot.
        folder_name: Folder under which the plots will be saved.
        hp: Optimizer hyperparameter to optimize over. Defaults to "lr"
        step: Number of steps to plot up to. If None, plot all steps.
    """
    ensure_all_exps_have_hyperparameter(exps, hp)
    problem = ensure_all_exps_have_same_problem(exps)

    exps_data = load_wandb_results(exps)
    exps_data = truncate_runs(exps_data, step)

    folder = Config.get_plots_directory() / folder_name / "grid"
    os.makedirs(folder, exist_ok=True)

    for metric, tr_va, logx, logy in itertools.product(
        problem.metrics, ["tr", "va"], [True, False], [True, False]
    ):
        if not metric.is_scalar():
            continue
        key = f"{tr_va}_{metric.__class__.__name__}"
        fig = make_step_size_grid_for_metric(exps_data, hp, key, (logx, logy))
        save_and_close(fig, folder, [key, f"{scale_str(logx)}x", f"{scale_str(logy)}y"])


def make_step_size_grid_for_metric(
    exps_data: Dict[Experiment, DataFrame],
    hp: str,
    metric_key: str,
    log_x_y: Tuple[bool, bool] = (False, False),
) -> plt.Figure:

    fig, ax = make_axes(plt, rel_width=1.0, nrows=1, ncols=1)

    optim_groups = group_experiments_by_optimizers(exps_data.keys(), hp)
    for i, (optim, exps) in enumerate(optim_groups.items()):
        group_exps_data = {exp: exps_data[exp] for exp in exps}
        hps, metrics = get_hp_and_metrics_at_end_per_hp(group_exps_data, hp, metric_key)
        ax.fill_between(
            hps,
            [np.min(sanitize(metrics[hp])) for hp in hps],
            [np.max(sanitize(metrics[hp])) for hp in hps],
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

    set_ylimits_to_fit_data_range(ax, exps_data, metric_key, log_x_y[1])
    hp_values = [getattr(exp.optim, hp) for exp in exps_data.keys()]
    set_limits(ax, x_y="x", limits=(min(hp_values), max(hp_values)), log=log_x_y[0])
    set_scale(ax, log_x_y)

    ax.set_title(f"Grid for {metric_key}")
    ax.set_xlabel(hp)
    ax.set_ylabel(metric_key)
    ax.legend()

    return fig
