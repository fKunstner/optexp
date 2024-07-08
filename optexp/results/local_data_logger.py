from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pandas as pd

from optexp import config
from optexp.config import get_logger
from optexp.experiment import Experiment
from optexp.results.data_logger import DataLogger
from optexp.results.rate_limited_logger import RateLimitedLogger
from optexp.results.utils import flatten_dict, pprint_dict


class LocalDataLogger(DataLogger):
    def __init__(self, experiment: Experiment, start_time: str) -> None:
        self.exp = experiment
        if not os.path.exists(local_save_dir(self.exp)):
            os.makedirs(local_save_dir(self.exp))

        self.console_logger = RateLimitedLogger(time_interval=1)
        experiment3 = self.exp
        self.handler = config.set_logfile(
            local_save_dir(experiment3) / f"{start_time}.log"
        )

        self._current_dict: dict[str, float | list[float]] = {}
        self._dicts: list[dict[str, float | list[float]]] = []

    def log_data(self, metric_dict: dict) -> None:
        self._current_dict.update(metric_dict)

    def commit(self) -> None:
        self.console_logger.log(pprint_dict(self._current_dict))
        self._dicts.append(copy.deepcopy(self._current_dict))
        self._current_dict = {}

    def finish(self, exit_code, stopped_early) -> None:
        experiment = self.exp
        filepath_csv = local_save_dir(experiment) / "data.csv"
        experiment1 = self.exp
        filepath_json = local_save_dir(experiment1) / "config.json"

        get_logger().info(f"Saving experiment configs to {filepath_json}")
        config_dict = flatten_dict(self.exp.loggable_dict())
        with open(filepath_json, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=4, sort_keys=True)

        get_logger().info(f"Saving experiment results to {filepath_csv}")
        pd.DataFrame.from_records(self._dicts).to_csv(filepath_csv)

        config.remove_loghandler(handler=self.handler)


def load_local_results(experiment: Experiment) -> pd.DataFrame:
    """Load the local results of an experiment."""
    filepath_csv = local_save_dir(experiment) / "data.csv"
    if not os.path.exists(filepath_csv):
        raise FileNotFoundError(f"File not found: {filepath_csv}")
    return pd.read_csv(filepath_csv)


def local_save_dir(exp: Experiment) -> Path:
    return config.get_hash_directory(
        config.get_experiment_directory(),
        exp.equivalent_hash(),
        exp.equivalent_definition(),
    )
