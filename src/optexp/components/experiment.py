import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from optexp import config
from optexp.components.component import Component
from optexp.components.hardwareconfigs.hardwareconfig import HardwareConfig
from optexp.components.optimizers.optimizer import Optimizer
from optexp.components.problem import Problem


@dataclass(frozen=True)
class Experiment(Component):
    """Represents an experiments where a problem is optimized given an optimizer."""

    # pylint: disable=too-many-instance-attributes
    optim: Optimizer
    problem: Problem
    group: str
    eval_every: int
    seed: int
    steps: int
    hw_config: HardwareConfig

    def exp_id(self) -> str:
        """Return a unique identifier for this experiment.

        Not a unique identifier for the current run of the experiments. Is unique for
        the definition of the experiments, combining the problem, optimizer, and seed.
        """
        return hashlib.sha1(str.encode(repr(self))).hexdigest()

    def save_directory(self) -> Path:
        """Return the directory where the experiments results are saved."""
        base = config.get_experiment_directory()
        exp_dir = (
            f"{self.problem.__class__.__name__}_"
            f"{self.problem.model.__class__.__name__}_"
            f"{self.problem.dataset.__class__.__name__}"
        )
        save_dir = base / exp_dir / self.exp_id()
        return save_dir

    def load_data(self):
        """Tries to load any data for the experiments.

        Starts by trying to load data from the wandb download folder, if that fails it
        tries to load data from the local runs folder.
        """
        try:
            df = self._load_wandb_data()
        except FileNotFoundError:
            print(f"Experiment did not have wandb data for, trying local data [{self}]")
            df = self._load_local_data()
        return df

    def _load_local_data(self) -> Optional[pd.DataFrame]:
        """Loads the most recent experiments run data saved locally."""
        save_dir = self.save_directory()
        # get the timestamps of the runs from the names of the files
        time_stamps = [
            time.strptime(str(Path(x).stem), "%Y-%m-%d--%H-%M-%S")
            for x in os.listdir(save_dir)
        ]

        if time_stamps is None:
            return None

        most_recent_run = max(time_stamps)
        csv_file_path = (
            save_dir / f"{time.strftime('%Y-%m-%d--%H-%M-%S', most_recent_run)}.csv"
        )
        run_data = pd.read_csv(csv_file_path)
        return run_data

    def _load_wandb_data(self) -> pd.DataFrame:
        """Loads data from most recent run of experiments from wandb."""
        save_dir = (
            config.get_wandb_cache_directory()
            / Path(self.group)
            / f"{self.exp_id()}.parquet"
        )
        if not save_dir.is_file():
            raise FileNotFoundError(f"File not found for experiment {self}")

        run_data = pd.read_parquet(save_dir)
        return run_data
