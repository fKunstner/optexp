from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

import optexp.config
from optexp.component import Component
from optexp.hardwareconfig import StrictManualConfig
from optexp.hardwareconfig.hardwareconfig import HardwareConfig
from optexp.optim.optimizer import Optimizer
from optexp.problem import Problem


@dataclass(frozen=True)
class Experiment(Component):
    """Specify an experiment.

    Args:
         optim (Optimizer): optimizer to use.
         problem (Problem): problem to solve.
         eval_every (int): often to evaluate the metrics.
         steps (int): total number of steps.
            To convert from epochs, use :func:`optexp.utils.epochs_to_steps`.
         seed (int, optional): seed for the random number generator.
            Defaults to 0.
         hardware_config (HardwareConfig, optional): implementation details.
            Defaults to :class:`~optexp.hardwareconfig.StrictManualConfig()`.
         group (str, optional): name for logging. Defaults to ``"default"``.
    """

    optim: Optimizer
    problem: Problem
    eval_every: int
    steps: int
    seed: int = 0
    hardware_config: HardwareConfig = field(
        default=StrictManualConfig(), repr=False, hash=False
    )
    group: str = field(default="default", repr=False, hash=False)

    def local_save_directory(self):
        """Return the directory where the experiments results are saved."""
        return optexp.config.get_hash_directory(
            optexp.config.get_experiment_directory(),
            self.equivalent_hash(),
            self.equivalent_definition(),
        )

    def wandb_download_directory(self):
        """Return the directory where the experiments results will be downloaded to."""
        return optexp.config.get_hash_directory(
            optexp.config.get_wandb_cache_directory(),
            self.equivalent_hash(),
            self.equivalent_definition(),
        )

    def load_data(self) -> Optional[pd.DataFrame]:
        """Tries to load any results for the experiments."""
        parquet_file = (
            self.wandb_download_directory() / f"{self.short_equivalent_hash()}.parquet"
        )
        if not parquet_file.is_file():
            optexp.config.get_logger().warning(
                f"No results found for experiment {self.short_equivalent_hash()} "
                f"in wandb download folder. Full experiment: {self}"
            )
            return None
        return pd.read_parquet(parquet_file)
