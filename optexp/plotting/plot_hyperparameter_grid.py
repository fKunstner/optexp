import itertools
import os
from typing import List, Optional

import attrs
import matplotlib.pyplot as plt

from optexp.config import Config
from optexp.experiment import Experiment
from optexp.optim import Optimizer
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

    for exp in exps:
        exps_data[exp] = exps_data[exp].loc[:up_to_step]

    for metric, logx, logy in itertools.product(
        problem.metrics, [True, False], [True, False]
    ):
        fig = make_step_size_grid_for_metric(
            exps_data, hyperparameter, metric, logx, logy
        )

        save_dir = Config.get_plots_directory() / folder_name
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_dir / f"{metric}_logx_{logx}_logy_{logy}.png")
        plt.close(fig)


def get_optimizers_groups(
    exps: List[Experiment], hyperparameter="lr"
) -> List[List[Optimizer]]:
    """Group optimizers that differ only in hyperparameter together.

    Args:
        exps: List of experiments to group by optimizer
        hyperparameter: Name of the Optimizer parameter to group together

    Returns:
        List of groups of optimizers that are the same, except for the give hyperparameter.
        For example:

            groups = get_optimizers_groups([
                SGD(lr=0.1, momentum=0),
                SGD(lr=1.0, momentum=0),
                SGD(lr=0.1, momentum=0.9),
                SGD(lr=1.0, momentum=0.9),
            ], hyperparameter="lr")
            groups == [
                [SGD(lr=0.1, momentum=0), SGD(lr=1.0, momentum=0)],
                [SGD(lr=0.1, momentum=0.9), SGD(lr=1.0, momentum=0.9)],
            ]
    """
    groups: List[List[Optimizer]] = []
    for exp in exps:
        for group in groups:
            step_size_in_group = getattr(group[0], hyperparameter)
            if attrs.evolve(exp.optim, **{hyperparameter: step_size_in_group}) in group:
                group.append(exp.optim)
                break
        groups.append([exp.optim])
    return groups


def make_step_size_grid_for_metric(
    exps_data, hyperparameter, metric, logx, logy  #  pylint: disable=unused-argument
) -> plt.Figure:
    """
    exps = exps_data.keys()

    optim_groups = get_optimizers_groups(exps, hyperparameter)
    fig, axes = make_axes(plt, rel_width=1.0, nrows=1, ncols=1)

    for optim_group in optim_groups:
        exp_group = [exp for exp in exps if exp.optim in optim_group]
    """

    raise NotImplementedError()
