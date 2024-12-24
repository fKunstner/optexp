import json
import subprocess
from pathlib import Path
from typing import Optional

import pandas as pd
import wandb
from tqdm import tqdm

from optexp.config import Config, get_logger
from optexp.experiment import Experiment
from optexp.results.data_logger import DataLogger
from optexp.results.utils import flatten_dict, get_hash_directory, numpyfy
from optexp.results.wandb_api import WandbAPI, get_wandb_runs

CONFIG_FILENAME = "config.json"
MISC_FILENAME = "misc.json"
DATA_FILENAME = "data.parquet"


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

        self.experiment = experiment
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
                self.save_info_for_syncing()

    def save_info_for_syncing(self):
        info = {
            "exp": self.experiment.equivalent_definition(),
            "exp_hash": self.experiment.equivalent_hash(),
            "folder": self._run_directory(),
        }
        basepath = Config.get_workspace_directory() / "syncing" / "to_sync"
        basepath.mkdir(parents=True, exist_ok=True)
        filename = self._run_directory().name + ".json"
        with open(basepath / filename, "w", encoding="utf-8") as f:
            json.dump(info, f)

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
    parquet_file = wandb_download_dir(exp) / DATA_FILENAME

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
    return (wandb_download_dir(exp) / DATA_FILENAME).exists()


def download_experiment(exp: Experiment):
    download_experiments([exp])


def download_experiments(exps: list[Experiment]) -> None:
    runs_for_exps = get_wandb_runs(exps)

    only_new_exps = [exp for exp in exps if not is_downloaded(exp)]

    print(
        f"New experiments to download: {len(only_new_exps)} "
        f"(already downloaded {len(exps)-len(only_new_exps)}/{len(exps)}"
    )

    for exp in tqdm(only_new_exps, total=len(only_new_exps)):
        runs = runs_for_exps[exp]
        if is_downloaded(exp):
            continue

        if len(runs) == 0:
            raise ValueError(
                f"No finished runs found for experiment {exp.short_equivalent_hash()}, "
                f"Has the experiment been uploaded? (full experiment: {exp})."
            )
        if len(runs) > 1:
            raise ValueError(
                f"Multiple finished runs found for experiment {exp.short_equivalent_hash()}. "
                f"Only one run is expected. Check the runs at \n"
                + "\n".join(
                    f"    https://wandb.ai/{WandbAPI.get_path()}/runs/{run.id}"
                    for run in runs
                )
                + f"\nfull experiment: {exp}"
            )

        run = runs[0]
        data_df = numpyfy(run.history(pandas=True, samples=10000))

        if data_df.empty:
            log_debug_info_empty_history(exp, run)
            continue

        download_dir = wandb_download_dir(exp)
        config_df = pd.DataFrame.from_records(flatten_dict(run.config))
        config_df.to_json(download_dir / CONFIG_FILENAME)

        linecount = run._attrs["historyLineCount"]  # pylint: disable=protected-access
        misc_df = pd.DataFrame.from_records(
            {
                "name": run.name,
                "id": run.id,
                "group": run.group,
                "state": run.state,
                "tags": run.tags,
                "histLineCount": linecount,
            }
        )
        misc_df.to_json(download_dir / MISC_FILENAME)

        data_df.to_parquet(download_dir / DATA_FILENAME)


def log_debug_info_empty_history(exp, run):

    cmd = find_sync_command_in_logs(run)
    how_to_sync = "Logs did not contain sync command."
    if cmd is not None:
        how_to_sync = f"The experiment was synced with `{cmd}`"

    run_url = "https://wandb.ai/" + "/".join(run.path)

    get_logger().warning(
        f"Experiment history is empty for experiment {exp.short_equivalent_hash()}. "
        "This might be due to a syncing issue. For additional information, see"
        f"  Wandb run: {run_url}"
        f"  Full experiment: {exp}"
        f"  {how_to_sync}"
    )


def find_sync_command_in_logs(run):
    run.file("output.log").download("/tmp/wandb", replace=True)
    with open("/tmp/wandb/output.log", "r", encoding="utf-8") as f:
        log_lines = f.readlines()
        for line in log_lines:
            if "wandb sync" in line:
                return line.strip()
    return None


def remove_experiments_that_are_already_saved(
    experiments: list[Experiment],
) -> list[Experiment]:
    """Checks a list of experiments against the experiments stored on wandb.

    Returns only the experiments that are not saved and marked as successful.
    """
    saved_experiments = get_wandb_runs(experiments)
    return [exp for exp in experiments if len(saved_experiments[exp]) == 0]
