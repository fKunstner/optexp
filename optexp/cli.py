import argparse
import os
import sys
from pathlib import Path
from shutil import rmtree
from typing import Dict, List, Optional

from attr import frozen
from tqdm import tqdm

from optexp.config import get_logger, use_wandb_config
from optexp.datasets.dataset import Downloadable
from optexp.experiment import Experiment
from optexp.plotting.plot_best import plot_metrics_over_time_for_best
from optexp.plotting.plot_hyperparameter_grid import plot_optim_hyperparam_grids
from optexp.results.wandb_data_logger import (
    download_experiments,
    remove_experiments_that_are_already_saved,
    wandb_download_dir,
)
from optexp.runner.runner import run_experiment
from optexp.runner.slurm.sbatch_writers import make_jobarray_file_contents
from optexp.runner.slurm.slurm_config import SlurmConfig
from optexp.utils import remove_duplicate_exps


@frozen
class ExpGroup:
    """A group of experiments to run together with a shared SlurmConfig."""

    exps: List[Experiment]
    slurm_config: Optional[SlurmConfig] = None


def cli(
    experiments: List[Experiment] | Dict[str, ExpGroup],
    slurm_config: Optional[SlurmConfig] = None,
    python_file: Optional[Path] = None,
) -> None:
    """Command line interface for running experiments.

    If the experiment file contains only one "group" of experiments,
        experiments is expected to be a list of Experiment objects
        to be run with the given SlurmConfig.

    If the experiment file contains multiple groups of experiments,
        experiments is expected to be a dictionary of ExpGroup objects
        containing the list of experiment and slurmconfig to run them with,
        indexed by group name.

    Args:
        experiments: List of experiments to run or a dictionary of groups to experiments.
        slurm_config: Configuration to use for running experiments on Slurm
        python_file: Path to the python file to run the experiments from.
            Defaults to the file that called this function, sys.argv[0].
    """

    exp_groups: Dict[str, ExpGroup] = (
        experiments
        if isinstance(experiments, dict)
        else {
            experiments[0].group: ExpGroup(exps=experiments, slurm_config=slurm_config)
        }
    )

    for group, exp_group in exp_groups.items():
        if not all(exp.group == group for exp in exp_group.exps):
            groupnames_in_exps = {exp.group for exp in exp_group.exps}
            raise ValueError(
                f"All experiments in group {group} must have the same group name. "
                f"Got {groupnames_in_exps} in group {group}."
            )

    available_groups = list(exp_groups.keys())
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-g",
        "--group",
        type=str,
        help="Group to run experiments from",
        choices=available_groups,
        default=None,
    )
    args = make_subparsers(parser).parse_args()

    groups_to_run = [args.group] if args.group is not None else available_groups
    for group_to_run in groups_to_run:
        print("Running group", group_to_run)
        exp_group = experiments[group_to_run]
        args.func(
            args,
            experiments=remove_duplicate_exps(exp_group.exps),
            group=group_to_run,
            slurm_config=exp_group.slurm_config,
            python_file=python_file,
        )


def make_subparsers(parser):
    subparsers = parser.add_subparsers(required=True)
    make_run_parser(subparsers)
    make_check_parser(subparsers)
    make_download_parser(subparsers)
    make_prepare_parser(subparsers)
    make_plot_parser(subparsers)
    return parser


def make_run_parser(subparsers):
    run_parser = subparsers.add_parser("run", help="Run the experiments")
    run_command = run_parser.add_mutually_exclusive_group(required=True)
    run_command.add_argument(
        "--local",
        action="store_true",
        help="Run experiments locally.",
        default=False,
    )
    run_command.add_argument(
        "--slurm",
        action="store_true",
        help="Run experiments on slurm.",
        default=False,
    )
    run_command.add_argument(
        "--single",
        action="store",
        type=int,
        help="Run a single experiment locally by index.",
        default=None,
    )
    run_modifier = run_parser.add_mutually_exclusive_group()
    run_modifier.add_argument(
        "--test",
        action="store_true",
        help="Run the first experiment. Alias for --single 0.",
        default=False,
    )
    run_parser.add_argument(
        "-f",
        "--force-rerun",
        action="store_true",
        help="Force rerun of experiments that are already saved.",
        default=False,
    )
    wandb_modifier = run_parser.add_mutually_exclusive_group()
    wandb_modifier.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb",
        default=None,
    )
    wandb_modifier.add_argument(
        "--wandb",
        action="store_true",
        help="Enable wandb",
        default=None,
    )
    autosync_modifier = run_parser.add_mutually_exclusive_group()
    autosync_modifier.add_argument(
        "--no-autosync",
        action="store_true",
        help="Enable wandb",
        default=None,
    )
    autosync_modifier.add_argument(
        "--autosync",
        action="store_true",
        help="Enables autosync",
        default=None,
    )

    def run_handler(
        args,
        experiments: List[Experiment],
        group: str,
        slurm_config: Optional[SlurmConfig] = None,
        python_file: Optional[Path] = None,
    ):

        def resolve(should_enable: Optional[bool], should_disable) -> Optional[bool]:
            if should_enable and should_disable:
                raise ValueError("Cannot both enable and disable a flag")
            if should_enable is None:
                if should_disable is None:
                    return None
                return not should_disable
            return should_enable

        with use_wandb_config(
            enabled=(resolve(args.wandb, args.no_wandb)),
            autosync=(resolve(args.autosync, args.no_autosync)),
        ):
            if args.single is not None:
                idx = int(args.single)
                validate_index(experiments, idx)
                run_locally([experiments[idx]], force_rerun=args.force_rerun)
                return

            if args.test:
                idx = 0
                validate_index(experiments, idx)
                experiments = [experiments[idx]]

            if args.local:
                run_locally(experiments, force_rerun=args.force_rerun)
                return

            if args.slurm:
                slurm_config = validate_slurm_config(slurm_config)
                run_slurm(
                    experiments,
                    group,
                    slurm_config,
                    force_rerun=args.force_rerun,
                    python_file=python_file,
                )
                return

            run_parser.print_help()

    run_parser.set_defaults(func=run_handler)


def make_plot_parser(subparsers):
    plot_parser = subparsers.add_parser("plot", help="Make plots")
    plot_parser.add_argument(
        "--grid",
        action="store_true",
        help="Plot the performance vs. hyperparameter.",
        default=False,
    )
    plot_parser.add_argument(
        "--best",
        action="store_true",
        help="Plot the performance over time for the ``best'' hyperparameter",
        default=False,
    )
    plot_parser.add_argument(
        "--hyperparam",
        type=str,
        help="Name of the optimizer hyperparameter to use for --grid or --best. Defaults to 'lr'.",
        default="lr",
    )
    plot_parser.add_argument(
        "--step",
        type=int,
        help="Maximum number of steps used for plotting. Useful to truncate long runs.",
        default=None,
    )
    plot_parser.add_argument(
        "--best-metric",
        type=str,
        help=(
            "Metric to use to determine the ``best'' run. "
            "Defaults to the training loss. Use as ``tr_CrossEntropy''."
        ),
        default=None,
    )
    plot_parser.add_argument(
        "--regularized",
        action="store_true",
        help="Adds regularization to loss.Defaults false",
        default=False,
    )
    plot_parser.add_argument(
        "--folder",
        type=str,
        help="Folder to save the plots in. Defaults to the group name.",
        default=None,
    )

    def plot_handler(
        args,
        experiments: List[Experiment],
        group: str,
        slurm_config: Optional[SlurmConfig] = None,  # pylint: disable=unused-argument
        python_file: Optional[Path] = None,  # pylint: disable=unused-argument
    ):
        if not any([args.grid, args.best]):
            print("Plotting both grid and best")
            args.grid = True
            args.best = True

        if args.grid:
            folder_name = args.folder
            if folder_name is None:
                folder_name = group

            plot_optim_hyperparam_grids(
                experiments,
                folder_name=folder_name,
                hp=args.hyperparam,
                step=args.step,
            )
        if args.best:
            folder_name = args.folder
            if folder_name is None:
                folder_name = group
                if args.regularized and args.best_metric is not None:
                    raise ValueError(
                        f"Can only regularize tr_CrossEntropy loss. "
                        f"best_metric set to {args.best_metric}."
                    )

            plot_metrics_over_time_for_best(
                experiments,
                folder_name=folder_name,
                hp=args.hyperparam,
                regularized=args.regularized,
                step=args.step,
                metric_key=args.best_metric,
            )

    plot_parser.set_defaults(func=plot_handler)


def make_prepare_parser(subparsers):
    prepare_parser = subparsers.add_parser(
        "prepare", help="Download results from the experiments"
    )

    def prepare_handler(
        args,  # pylint: disable=unused-argument
        experiments: List[Experiment],
        group: str,  # pylint: disable=unused-argument
        slurm_config: Optional[SlurmConfig] = None,  # pylint: disable=unused-argument
        python_file: Optional[Path] = None,  # pylint: disable=unused-argument
    ):
        prepare(experiments)

    prepare_parser.set_defaults(func=prepare_handler)


def make_download_parser(subparsers):
    download_parser = subparsers.add_parser(
        "download", help="Download results from the experiments"
    )
    download_parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear cache of downloaded results from wandb",
        default=False,
    )

    def download_handler(
        args,
        experiments: List[Experiment],
        group: str,  # pylint: disable=unused-argument
        slurm_config: Optional[SlurmConfig] = None,  # pylint: disable=unused-argument
        python_file: Optional[Path] = None,  # pylint: disable=unused-argument
    ):
        if args.clear:
            clear_downloaded_data(experiments)
            return 0

        download_data(experiments)
        return 0

    download_parser.set_defaults(func=download_handler)
    return download_parser


def make_check_parser(subparsers):
    check_parser = subparsers.add_parser(
        "check", help="Check the status of the experiments"
    )

    def check_handler(
        args,  # pylint: disable=unused-argument
        experiments: List[Experiment],
        group: str,  # pylint: disable=unused-argument
        slurm_config: Optional[SlurmConfig] = None,  # pylint: disable=unused-argument
        python_file: Optional[Path] = None,  # pylint: disable=unused-argument
    ):
        report(experiments, None)
        return 0

    check_parser.set_defaults(func=check_handler)


def validate_index(experiments, idx):
    if not 0 <= idx < len(experiments):
        raise ValueError(
            f"Given index {idx} out of bounds for {len(experiments)} experiments"
        )


def validate_slurm_config(slurm_config) -> SlurmConfig:
    if slurm_config is None:
        raise ValueError("Must provide a SlurmConfig if running on slurm. Got None.")
    return slurm_config


def report(experiments: List[Experiment], by: Optional[str]) -> None:
    """Generate a report of what experiments have been run/are stored on wandb."""
    if by is not None:
        raise NotImplementedError

    all_exps = experiments
    remaining_exps = remove_experiments_that_are_already_saved(experiments)

    n = len(all_exps)
    n_missing = len(remaining_exps)
    percent_complete = ((n - n_missing) / n) * 100
    print(
        f"Out of {n} experiments, {n_missing} still need to run "
        f"({percent_complete:.2f}% complete)"
    )


def run_locally(experiments: List[Experiment], force_rerun: bool) -> None:
    """Run experiments locally."""
    get_logger().info(f"Preparing to run {len(experiments)} experiments")
    if not force_rerun:
        original_n_exps = len(experiments)
        experiments = remove_experiments_that_are_already_saved(experiments)
        get_logger().info(
            f"New experiments to run: {len(experiments)}/{original_n_exps}"
        )

    datasets = set(exp.problem.dataset for exp in experiments)

    for dataset in datasets:
        if isinstance(dataset, Downloadable) and not dataset.is_downloaded():
            dataset.download()

    for exp in tqdm(experiments):
        run_experiment(exp)


def run_slurm(
    experiments: List[Experiment],
    group: str,
    slurm_config: SlurmConfig,
    force_rerun: bool,
    python_file: Optional[Path] = None,
) -> None:
    """Run experiments on Slurm."""
    get_logger().info(
        f"Preparing experiments to run {len(experiments)} experiments on Slurm"
    )
    if python_file is None:
        path_to_python_script = Path(sys.argv[0]).resolve()
    else:
        path_to_python_script = python_file

    if not force_rerun:
        print("  Checking which experiments have to run")
        exps_to_run = remove_experiments_that_are_already_saved(experiments)
        should_run = [exp in exps_to_run for exp in experiments]
        print(f"    Should run {should_run.count(True)}/{len(should_run)} experiments")
    else:
        should_run = [True for _ in experiments]

    contents = make_jobarray_file_contents(
        experiment_file=path_to_python_script,
        group=group,
        should_run=should_run,
        slurm_config=slurm_config,
        force_rerun=force_rerun,
    )

    tmp_filename = f"tmp_{group}.sh"
    print(f"  Saving sbatch file in {tmp_filename}")
    with open(tmp_filename, "w+", encoding="utf-8") as file:
        file.writelines(contents)

    datasets = set(exp.problem.dataset for exp in experiments)

    for dataset in datasets:
        if isinstance(dataset, Downloadable):
            dataset.download()

    print("  Sending experiments to Slurm - executing sbatch file")
    os.system(f"sbatch {tmp_filename}")


def download_data(experiments: List[Experiment]) -> None:
    """Download results from experiments into wandb cache."""

    get_logger().info(
        f"Preparing to download results from {len(experiments)} experiments"
    )
    download_experiments(experiments)


def clear_downloaded_data(experiments: List[Experiment]) -> None:
    """Download results from experiments into wandb cache."""
    get_logger().info(f"Clearing cache for {len(experiments)} experiments")
    for exp in tqdm(experiments):
        path = wandb_download_dir(exp)
        if not os.path.exists(path):
            return
        rmtree(path)


def prepare(experiments):
    unique_datasets = set(exp.problem.dataset for exp in experiments)
    get_logger().info(f"Preparing results for {len(unique_datasets)} datasets")
    for dataset in unique_datasets:
        if isinstance(dataset, Downloadable) and not dataset.is_downloaded():
            get_logger().info(f"Downloading dataset {dataset}")
            dataset.download()
        get_logger().info(f"Dataset {dataset} is ready")
    get_logger().info("Done")
