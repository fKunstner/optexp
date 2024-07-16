import argparse
import os
import sys
from pathlib import Path
from shutil import rmtree
from typing import List, Optional

from tqdm import tqdm

from optexp.config import get_logger, use_wandb_config
from optexp.datasets.dataset import Downloadable
from optexp.experiment import Experiment
from optexp.results.wandb_data_logger import (
    download_experiments,
    remove_experiments_that_are_already_saved,
    wandb_download_dir,
)
from optexp.runner.runner import run_experiment
from optexp.runner.slurm.sbatch_writers import make_jobarray_file_contents
from optexp.runner.slurm.slurm_config import SlurmConfig
from optexp.utils import remove_duplicate_exps


def run_handler(
    args,
    experiments: List[Experiment],
    slurm_config: Optional[SlurmConfig] = None,
    python_file: Optional[Path] = None,
):

    def resolve(should_enable: Optional[bool], should_disable) -> Optional[bool]:
        if should_enable is None and should_disable is None:
            return None
        if should_enable and should_disable:
            raise ValueError("Cannot both enable and disable a flag")
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
                slurm_config,
                force_rerun=args.force_rerun,
                python_file=python_file,
            )
            return


def download_handler(
    args,
    experiments: List[Experiment],
    slurm_config: Optional[SlurmConfig] = None,  # pylint: disable=unused-argument
    python_file: Optional[Path] = None,  # pylint: disable=unused-argument
):
    if args.clear:
        clear_downloaded_data(experiments)
        return 0

    download_data(experiments)
    return 0


def check_handler(
    args,  # pylint: disable=unused-argument
    experiments: List[Experiment],
    slurm_config: Optional[SlurmConfig] = None,  # pylint: disable=unused-argument
    python_file: Optional[Path] = None,  # pylint: disable=unused-argument
):
    report(experiments, None)
    return 0


def prepare_handler(
    args,  # pylint: disable=unused-argument
    experiments: List[Experiment],
    slurm_config: Optional[SlurmConfig] = None,  # pylint: disable=unused-argument
    python_file: Optional[Path] = None,  # pylint: disable=unused-argument
):
    prepare(experiments)


def cli(
    experiments: List[Experiment],
    slurm_config: Optional[SlurmConfig] = None,
    python_file: Optional[Path] = None,
) -> None:
    """Command line interface for running experiments.

    Args:
        experiments: List of experiments to run
        slurm_config: Configuration to use for running experiments on Slurm
        python_file: Path to the python file to run the experiments from.
            Defaults to the file that called this function, sys.argv[0]
    """
    experiments = remove_duplicate_exps(experiments)

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)

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
    run_parser.set_defaults(func=run_handler)

    check_parser = subparsers.add_parser(
        "check", help="Check the status of the experiments"
    )
    check_parser.set_defaults(func=check_handler)

    download_parser = subparsers.add_parser(
        "download", help="Download results from the experiments"
    )
    download_parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear cache of downloaded results from wandb",
        default=False,
    )
    download_parser.set_defaults(func=download_handler)

    prepare_parser = subparsers.add_parser(
        "prepare", help="Download results from the experiments"
    )
    prepare_parser.set_defaults(func=prepare_handler)

    args = parser.parse_args()
    args.func(
        args,
        experiments=experiments,
        slurm_config=slurm_config,
        python_file=python_file,
    )


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
        if isinstance(dataset, Downloadable):
            dataset.download()

    for exp in tqdm(experiments):
        run_experiment(exp)


def run_slurm(
    experiments: List[Experiment],
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
        should_run=should_run,
        slurm_config=slurm_config,
    )

    group = experiments[0].group
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
