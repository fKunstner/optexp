import subprocess
from pathlib import Path
from typing import Optional

import pandas as pd
import wandb

from optexp.config import Config, get_logger
from optexp.experiment import Experiment
from optexp.results.data_logger import DataLogger
from optexp.results.utils import flatten_dict, get_hash_directory, numpyfy
from optexp.results.wandb_api import WandbAPI, get_wandb_runs

WANDB_CONFIG_FILENAME = "config.json"
WANDB_MISC_FILENAME = "misc.json"
WANDB_DATA_FILENAME = "data.parquet"


class WandbDataLogger(DataLogger):

    def __init__(
        self,
        experiment: Experiment,
        start_time: str,
    ) -> None:
        """Data logger for experiments.

        Delegates to a console logger to print progress.
        Saves the results to a csv and experiments configuration to a json file.
        Creates the save_dir if it does not exist.

        Args:
            experiment: The experiment to log.
        """

        if Config.get_wandb_key() is None:
            raise ValueError("WandB API key not set.")

        self.run = wandb.init(
            project=Config.get_wandb_project(),
            entity=Config.get_wandb_entity(),
            config={
                "short_equiv_hash": experiment.short_equivalent_hash(),
                "equiv_hash": experiment.equivalent_hash(),
                "equiv_def": experiment.equivalent_definition(),
                "start_time": start_time,
                "exp_config": experiment.loggable_dict(),
            },
            group=experiment.group,
            mode=Config.wandb_mode,
            dir=Config.get_experiment_directory(),
        )

        if self.run is None:
            raise ValueError("WandB run initialization failed.")

        get_logger().info(f"--- WANDB initialized. Wandb Run ID: {self.run.id}")
        get_logger().info(f"Sync with:\n {self._sync_command()}")

    def log(self, metric_dict: dict) -> None:
        """Log a dictionary of metrics.

        Based on the wandb log function (https://docs.wandb.ai/ref/python/log)
        Uses the concept of "commit" to separate different steps/iterations.

        log_data can be called multiple times per step,
        and repeated calls update the current logging dictionary.
        If metric_dict has the same keys as a previous call to log_data,
        the keys will get overwritten.

        To move on to the next step/iteration, call commit.

        Args:
            metric_dict: Dictionary of metrics to log
        """
        wandb.log(metric_dict, commit=False)

    def commit(self) -> None:
        wandb.log({}, commit=True)

    def finish(self, exit_code, stopped_early=False) -> None:
        if self.run is None:
            raise ValueError("Expected a WandB run but None found.")

        self.run.tags += ("finished",)
        if stopped_early:
            self.run.tags += ("stopped_early",)

        get_logger().info("Finishing Wandb run")
        wandb.finish(exit_code=exit_code)

        if Config.wandb_mode == "offline":
            if Config.wandb_autosync:
                get_logger().info(f"Uploading wandb run in {self._run_directory()}")
                get_logger().info("Sync with")
                get_logger().info(f"    {self._sync_command()}")
                subprocess.run(self._sync_command(), shell=True, check=False)
            else:
                get_logger().info("Not uploading run to wandb. To sync manually, run")
                get_logger().info(f"    {self._sync_command()}")

    def _run_directory(self):
        return Path(self.run.dir).parent

    def _sync_command(self):
        return f"wandb sync " f"{self._run_directory()}"


def load_wandb_results(exps: list[Experiment]) -> dict[Experiment, pd.DataFrame]:
    missing_experiments = list(filter(lambda exp: not is_downloaded(exp), exps))
    if len(missing_experiments) > 0:
        download_experiments(missing_experiments)
    return {exp: load_wandb_result(exp) for exp in exps}


def load_wandb_result(exp: Experiment) -> Optional[pd.DataFrame]:
    """Tries to load any results for the experiments."""
    parquet_file = wandb_download_dir(exp) / WANDB_DATA_FILENAME

    if not parquet_file.is_file():
        get_logger().warning(
            f"No results found for experiment {exp.short_equivalent_hash()} "
            f"in wandb download folder. Full experiment: {exp}"
        )
        return None
    return pd.read_parquet(parquet_file)


def wandb_download_dir(exp: Experiment) -> Path:
    return get_hash_directory(
        Config.get_wandb_cache_directory(),
        exp.equivalent_hash(),
        exp.equivalent_definition(),
    )


def is_downloaded(exp: Experiment) -> bool:
    return (wandb_download_dir(exp) / WANDB_DATA_FILENAME).exists()


def download_experiment(exp: Experiment):
    download_experiments([exp])


def download_experiments(exps: list[Experiment]) -> None:
    exp_to_runs = get_wandb_runs(exps)
    for exp, runs in exp_to_runs.items():
        if len(runs) == 0:
            raise ValueError(
                f"No finished runs found for experiment {exp.short_equivalent_hash()}"
                f"Has the experiment been uploaded? (full experiment: {exp})."
            )
        if len(runs) > 1:
            raise ValueError(
                f"Multiple finished runs found for experiment {exp.short_equivalent_hash()}."
                f"Only one run is expected. Check the runs at "
                + str(
                    f"https://wandb.ai/{WandbAPI.get_path()}/runs/{run.id}"
                    for run in runs
                )
                + f"(full experiment: {exp})."
            )

        run = runs[0]

        save_file = wandb_download_dir(exp) / WANDB_DATA_FILENAME
        if save_file.exists():
            raise FileExistsError(f"File already exists: {save_file}")

        pd.DataFrame.from_records(flatten_dict(run.config)).to_json(
            wandb_download_dir(exp) / WANDB_CONFIG_FILENAME
        )

        linecount = run._attrs["historyLineCount"]  # pylint: disable=protected-access
        pd.DataFrame.from_records(
            {
                "name": run.name,
                "id": run.id,
                "group": run.group,
                "state": run.state,
                "tags": run.tags,
                "histLineCount": linecount,
            }
        ).to_json(wandb_download_dir(exp) / WANDB_MISC_FILENAME)

        numpyfy(run.history(pandas=True, samples=10000)).to_parquet(
            wandb_download_dir(exp) / WANDB_DATA_FILENAME
        )


def remove_experiments_that_are_already_saved(
    experiments: list[Experiment],
) -> list[Experiment]:
    """Checks a list of experiments against the experiments stored on wandb.

    Returns only the experiments that are not saved and marked as successful.
    """
    saved_experiments = get_wandb_runs(experiments)
    return [exp for exp in experiments if len(saved_experiments[exp]) == 0]
