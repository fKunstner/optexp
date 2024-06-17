from __future__ import annotations

import copy
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import wandb

from optexp import config
from optexp.config import get_logger
from optexp.loggers.rate_limited_logger import RateLimitedLogger
from optexp.loggers.utils import pprint_dict


class DataLogger:
    def __init__(
        self,
        config_dict: Dict,
        group: str,
        run_id: str,
        exp_id: str,
        save_directory: Path,
        use_wandb: Optional[bool] = None,
        wandb_autosync: bool = True,
    ) -> None:
        """Data logger for experiments.

        Delegates to a console logger to print progress.
        Saves the data to a csv and experiments configuration to a json file.
        Creates the save_dir if it does not exist.

        Args:
            run_id: Unique id for the run
                (an experiments might have multiple runs)
            experiments: The experiments to log
        """

        self.wandb_autosync = wandb_autosync
        self.console_logger = RateLimitedLogger()

        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        self.handler = config.set_logfile(save_directory / f"{run_id}.log")
        self.use_wandb = config.get_wandb_status() if use_wandb is None else use_wandb

        if self.use_wandb:
            get_logger().info("WandB is enabled")
            if config.get_wandb_key() is not None:
                self.run = wandb.init(
                    project=config.get_wandb_project(),
                    entity=config.get_wandb_entity(),
                    config={
                        "exp_id": exp_id,
                        "run_id": run_id,
                        "exp_config": config_dict,
                    },
                    group=group,
                    mode=config.get_wandb_mode(),
                    dir=config.get_experiment_directory(),
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

    def finish(self, exit_code) -> None:
        """Save the results."""

        if self.use_wandb:
            if self.run is None:
                raise ValueError("Expected a WandB run but None found.")

            get_logger().info("Finishing Wandb run")
            wandb.finish(exit_code=exit_code)

            if config.get_wandb_mode() == "offline" and self.wandb_autosync:
                get_logger().info(f"Uploading wandb run in {Path(self.run.dir).parent}")
                get_logger().info(f"Sync with")
                get_logger().info(f"    {self._sync_command()}")
                subprocess.run(self._sync_command(), shell=True, check=False)
            else:
                get_logger().info("Not uploading run to wandb. To sync manually, run")
                get_logger().info(f"    {self._sync_command()}")

        config.remove_loghandler(handler=self.handler)

    def _sync_command(self):
        return f"wandb sync " f"{Path(self.run.dir).parent}"
