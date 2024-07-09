from __future__ import annotations

import time

from optexp import config
from optexp.config import get_logger
from optexp.experiment import Experiment
from optexp.results.data_logger import DataLogger
from optexp.results.local_data_logger import LocalDataLogger
from optexp.results.wandb_data_logger import WandbDataLogger


class MainDataLogger(DataLogger):
    def __init__(self, experiment: Experiment) -> None:
        start_time = time.strftime("%Y-%m-%d--%H-%M-%S")
        self.sub_data_loggers: list[DataLogger] = []
        if config.should_use_wandb():
            get_logger().info("WandB is enabled.")
            self.sub_data_loggers.append(WandbDataLogger(experiment, start_time))
        else:
            get_logger().warning("WandB is disabled.")

        self.sub_data_loggers.append(LocalDataLogger(experiment, start_time))

    def log(self, metric_dict: dict[str, float | list[float]]) -> None:
        for sub_data_logger in self.sub_data_loggers:
            sub_data_logger.log(metric_dict)

    def commit(self) -> None:
        for sub_data_logger in self.sub_data_loggers:
            sub_data_logger.commit()

    def finish(self, exit_code, stopped_early) -> None:
        for sub_data_logger in self.sub_data_loggers:
            sub_data_logger.finish(exit_code, stopped_early)
