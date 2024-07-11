import logging
import os
from contextlib import contextmanager
from logging import Logger
from pathlib import Path
from typing import Literal, Optional

import torch

NAME_WORKSPACE = "OPTEXP_WORKSPACE"
NAME_LOGLEVEL = "OPTEXP_CONSOLE_LOGGING_LEVEL"
NAME_WANDB_ENABLED = "OPTEXP_WANDB_ENABLED"
NAME_WANDB_PROJECT = "OPTEXP_WANDB_PROJECT"
NAME_WANDB_ENTITY = "OPTEXP_WANDB_ENTITY"
NAME_WANDB_MODE = "OPTEXP_WANDB_MODE"
NAME_WANDB_AUTOSYNC = "OPTEXP_WANDB_AUTOSYNC"
NAME_WANDB_API_KEY = "WANDB_API_KEY"
NAME_SLURM_EMAIL = "OPTEXP_SLURM_NOTIFICATION_EMAIL"
NAME_SLURM_ACCOUNT = "OPTEXP_SLURM_ACCOUNT"
NAME_SLURM_INSTALL_SCRIPT = "OPTEXP_SLURM_LOCAL_INSTALL_SCRIPT"
LOG_FMT = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"


CONSOLE_LOGGING_LEVEL = "INFO"


def get_logger(name: Optional[str] = None, level: Optional[str | int] = None) -> Logger:
    """Get a logger with a console handler.

    Args:
        name: Name of the logger.
        level: Logging level.
            Defaults to the value of the env variable OPTEXP_CONSOLE_LOGGING_LEVEL.
    """
    logger = logging.getLogger(__name__ if name is None else name)

    if not any(isinstance(x, logging.StreamHandler) for x in logger.handlers):
        sh = logging.StreamHandler()
        sh.setLevel(level=CONSOLE_LOGGING_LEVEL if level is None else level)
        formatter = logging.Formatter(LOG_FMT, datefmt="%Y-%m-%d %H:%M:%S")
        sh.setFormatter(formatter)
        logger.addHandler(sh)
        logger.setLevel(level=CONSOLE_LOGGING_LEVEL if level is None else level)
    return logger


def get_env_var(
    name: str, accepted: Optional[list] = None, default=None, converter=None
):

    def convert(x):
        if converter is None:
            return x
        return converter(x)

    if name in os.environ:
        val = os.environ[name]
        get_logger().debug(f"Reading environment variable {name}={val}")
        if accepted is not None and val not in accepted:
            raise ValueError(
                f"Invalid value for environment variable {name}. "
                f"Got {val}. Expected one of {accepted}."
            )
        return convert(os.environ[name])
    get_logger().debug(f"Environment variable {name} undefined. Defaults to {default}.")
    return convert(default)


class Config:
    wandb_timeout: int = 60
    wandb_enabled: bool = get_env_var(
        NAME_WANDB_ENABLED,
        accepted=["true", "false"],
        default="false",
        converter=lambda x: x.lower() == "true",
    )
    wandb_autosync: Optional[bool] = get_env_var(
        NAME_WANDB_AUTOSYNC,
        accepted=["true", "false"],
        default="false",
        converter=lambda x: x.lower() == "true",
    )
    wandb_entity: Optional[str] = get_env_var(
        NAME_WANDB_ENTITY,
        default=None,
    )
    wandb_api_key: Optional[str] = get_env_var(NAME_WANDB_API_KEY, default=None)
    slurm_account: Optional[str] = get_env_var(
        NAME_SLURM_ACCOUNT,
        default=None,
    )
    slurm_email: Optional[str] = get_env_var(
        NAME_SLURM_EMAIL,
        default=None,
    )
    workspace_directory: Optional[str] = get_env_var(
        NAME_WORKSPACE,
        default=None,
    )
    wandb_mode: Optional[Literal["online", "offline", "disabled"]] = get_env_var(
        NAME_WANDB_MODE,
        accepted=["online", "offline", "disabled"],
        default="offline",
    )
    wandb_project: Optional[str] = get_env_var(
        NAME_WANDB_PROJECT,
        default=None,
    )
    device: Literal["cpu", "cuda"] = "cuda" if torch.cuda.is_available() else "cpu"
    slurm_local_install_script: Optional[str] = get_env_var(
        NAME_SLURM_INSTALL_SCRIPT,
        default=None,
    )

    @staticmethod
    def get_slurm_account() -> str:
        if Config.slurm_account is None:
            raise ValueError(
                f"Slurm Account not set. Define the {NAME_SLURM_ACCOUNT} environment variable."
            )
        return Config.slurm_account

    @staticmethod
    def get_slurm_local_install_script_path() -> Path:
        if Config.slurm_local_install_script is None:
            raise ValueError(
                f"Slurm Install Script not set. Define the {NAME_SLURM_INSTALL_SCRIPT} environment variable."
            )
        if not Path(Config.slurm_local_install_script).exists():
            raise ValueError(
                f"Slurm Install Script not found. File {Config.slurm_local_install_script} does not exist."
            )
        return Path(Config.slurm_local_install_script)

    @staticmethod
    def get_workspace_directory() -> Path:
        if Config.workspace_directory is None:
            raise ValueError(
                f"Workspace not set. Define the {NAME_WORKSPACE} environment variable "
                "to set where to save datasets and experiments results."
            )
        return Path(Config.workspace_directory)

    @staticmethod
    def get_wandb_key() -> str:
        if Config.wandb_api_key is None:
            raise ValueError(
                f"WandB API key is not defined. Define the {NAME_WANDB_API_KEY} "
                "environment variable to set the API key"
            )
        return Config.wandb_api_key

    @staticmethod
    def get_wandb_project() -> str:
        if Config.wandb_project is None:
            raise ValueError(
                f"WandB project not set. Define the {NAME_WANDB_PROJECT} "
                "environment variable to define the project"
            )
        return Config.wandb_project

    @staticmethod
    def get_wandb_entity() -> str:
        if Config.wandb_entity is None:
            raise ValueError(
                f"WandB entity not set. Define the {NAME_WANDB_ENTITY} "
                "environment variable to define the entity"
            )
        return str(Config.wandb_entity)

    @staticmethod
    def get_dataset_directory() -> Path:
        return Config.get_workspace_directory() / "datasets"

    @staticmethod
    def get_tokenizers_directory() -> Path:
        return Config.get_workspace_directory() / "tokenizers"

    @staticmethod
    def get_experiment_directory() -> Path:
        return Config.get_workspace_directory() / "experiments"

    @staticmethod
    def get_wandb_cache_directory() -> Path:
        return Config.get_workspace_directory() / "wandb_cache"

    @staticmethod
    def get_device() -> Literal["cpu", "cuda"]:
        if Config.device != "cuda":
            get_logger().warning("GPU not available, running experiments on CPU.")
        return Config.device


@contextmanager
def use_wandb_config(
    enabled: Optional[bool] = None,
    autosync: Optional[bool] = None,
    project: Optional[str] = None,
    entity: Optional[str] = None,
    mode: Optional[Literal["online", "offline", "disabled"]] = None,
):
    previous_enabled = Config.wandb_enabled
    previous_autosync = Config.wandb_autosync
    previous_project = Config.wandb_project
    previous_entity = Config.wandb_entity
    previous_mode = Config.wandb_mode

    if enabled is not None:
        Config.wandb_enabled = enabled
    if autosync is not None:
        Config.wandb_autosync = autosync
    if project is not None:
        Config.wandb_project = project
    if entity is not None:
        Config.wandb_entity = entity
    if mode is not None:
        Config.wandb_mode = mode

    yield

    Config.wandb_enabled = previous_enabled
    Config.wandb_autosync = previous_autosync
    Config.wandb_project = previous_project
    Config.wandb_entity = previous_entity
    Config.wandb_mode = previous_mode


def set_logfile(path: Path, name: Optional[str] = None):
    handler = logging.FileHandler(path)
    handler.formatter = logging.Formatter(LOG_FMT)
    get_logger(name=name).addHandler(handler)
    return handler


def remove_loghandler(handler: logging.FileHandler, name: Optional[str] = None):
    get_logger(name=name).removeHandler(handler)
