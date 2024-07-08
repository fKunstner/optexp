from __future__ import annotations

import os
import subprocess
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import wandb

import optexp.config
from optexp.config import get_logger
from optexp.experiment import Experiment
from optexp.results.rate_limited_logger import RateLimitedLogger


class DataLogger(ABC):
    @abstractmethod
    def log_data(self, metric_dict: dict) -> None:
        pass

    @abstractmethod
    def commit(self) -> None:
        pass

    @abstractmethod
    def finish(self, exit_code, stopped_early) -> None:
        pass


class WandbDataLogger(DataLogger):

    def __init__(
        self,
        experiment: Experiment,
        use_wandb: Optional[bool] = None,
        wandb_autosync: Optional[bool] = None,
    ) -> None:
        """Data logger for experiments.

        Delegates to a console logger to print progress.
        Saves the results to a csv and experiments configuration to a json file.
        Creates the save_dir if it does not exist.

        Args:
            experiment: The experiment to log.
            use_wandb: Whether to use wandb for storing logs.
                Overrides the global :py:meth:`Config.should_use_wandb`
            wandb_autosync: Whether to call `wandb sync` at the end of the run.
                Overrides the global :py:meth:`Config.should_wandb_autosync`
        """

        if use_wandb is None:
            use_wandb = optexp.config.should_use_wandb()
        if wandb_autosync is None:
            wandb_autosync = optexp.config.should_wandb_autosync()

        save_directory = experiment.local_save_directory()
        start_time = time.strftime("%Y-%m-%d--%H-%M-%S")

        self.wandb_autosync = wandb_autosync
        self.console_logger = RateLimitedLogger()

        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        self.handler = optexp.config.set_logfile(save_directory / f"{start_time}.log")
        self.use_wandb = (
            optexp.config.should_use_wandb() if use_wandb is None else use_wandb
        )

        if self.use_wandb:
            get_logger().info("WandB is enabled")
            if optexp.config.get_wandb_key() is not None:
                self.run = wandb.init(
                    project=optexp.config.get_wandb_project(),
                    entity=optexp.config.get_wandb_entity(),
                    config={
                        "short_equiv_hash": experiment.short_equivalent_hash(),
                        "equiv_hash": experiment.equivalent_hash(),
                        "start_time": start_time,
                        "exp_config": experiment.loggable_dict(),
                    },
                    group=experiment.group,
                    mode=optexp.config.get_wandb_mode(),
                    dir=optexp.config.get_experiment_directory(),
                )
            else:
                raise ValueError("WandB API key not set.")
            if self.run is None:
                raise ValueError("WandB run initialization failed.")

            get_logger().info(f"--- WANDB initialized. Wandb Run ID: {self.run.id}")
            get_logger().info(f"Sync with:\n {self._sync_command()}")
        else:
            get_logger().info("WandB is NOT enabled.")

    def log_data(self, metric_dict: dict) -> None:
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
        if self.use_wandb:
            wandb.log(metric_dict, commit=False)

    def commit(self) -> None:
        """Commit the current logs and move on to the next step/iteration."""
        if self.use_wandb:
            wandb.log({}, commit=True)

    def finish(self, exit_code, stopped_early=False) -> None:
        """Save the results."""

        if self.use_wandb:
            if self.run is None:
                raise ValueError("Expected a WandB run but None found.")

            self.run.tags += ("finished",)
            if stopped_early:
                self.run.tags += ("stopped_early",)

            get_logger().info("Finishing Wandb run")
            wandb.finish(exit_code=exit_code)

            if optexp.config.get_wandb_mode() == "offline" and self.wandb_autosync:
                get_logger().info(f"Uploading wandb run in {Path(self.run.dir).parent}")
                get_logger().info("Sync with")
                get_logger().info(f"    {self._sync_command()}")
                subprocess.run(self._sync_command(), shell=True, check=False)
            else:
                get_logger().info("Not uploading run to wandb. To sync manually, run")
                get_logger().info(f"    {self._sync_command()}")

        optexp.config.remove_loghandler(handler=self.handler)

    def _sync_command(self):
        return f"wandb sync " f"{Path(self.run.dir).parent}"


class DummyDataLogger(DataLogger):
    """A dummy results logger that does nothing."""

    def __init__(self, experiment: Optional[Experiment] = None) -> None:
        pass

    def log_data(self, metric_dict: dict) -> None:
        pass

    def commit(self) -> None:
        pass

    def finish(self, exit_code, stopped_early) -> None:
        pass
