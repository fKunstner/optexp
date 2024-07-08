import json
import logging
import os
from logging import Logger
from pathlib import Path
from typing import Literal, Optional

import torch

ENV_VAR_WORKSPACE = "OPTEXP_WORKSPACE"
ENV_VAR_LOGGING = "OPTEXP_CONSOLE_LOGGING_LEVEL"
ENV_VAR_WANDB_ENABLED = "OPTEXP_WANDB_ENABLED"
ENV_VAR_WANDB_PROJECT = "OPTEXP_WANDB_PROJECT"
ENV_VAR_WANDB_ENTITY = "OPTEXP_WANDB_ENTITY"
ENV_VAR_WANDB_MODE = "OPTEXP_WANDB_MODE"
ENV_VAR_WANDB_AUTOSYNC = "OPTEXP_WANDB_AUTOSYNC"
ENV_VAR_WANDB_API_KEY = "WANDB_API_KEY"
ENV_VAR_SLURM_EMAIL = "OPTEXP_SLURM_NOTIFICATION_EMAIL"
ENV_VAR_SLURM_ACCOUNT = "OPTEXP_SLURM_ACCOUNT"
LOG_FMT = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"


class _CONFIG:
    should_use_wandb = None


class DisableWandb:
    def __init__(self):
        self.wandb_status = should_use_wandb()

    def __enter__(self):
        _CONFIG.should_use_wandb = False

    def __exit__(self, *args, **kwargs):
        _CONFIG.should_use_wandb = self.wandb_status


class UseWandbProject:
    """Context manager to set the wandb project for a block of code.

    Temporarily overrides the global project set in the environment variable. Used in
    get_wandb_project().
    """

    global_project: Optional[str] = None

    def __init__(self, project: Optional[str] = None):
        self.project_for_context = project
        self.project_outside_context = None

    def __enter__(self):
        self.project_outside_context = self.global_project
        UseWandbProject.global_project = self.project_for_context

    def __exit__(self, *args, **kws):
        UseWandbProject.global_project = self.project_outside_context


def get_wandb_key() -> str:
    api_key = os.environ.get(ENV_VAR_WANDB_API_KEY, None)
    if api_key is None:
        raise ValueError(
            f"WandB API key is not defined. Define the {ENV_VAR_WANDB_API_KEY} "
            "environment variable to set the API key"
        )
    return api_key


def get_wandb_timeout() -> int:
    """Timeout for results transfers, in seconds.

    Large timeout are needed to download runs with large logs (per class).
    """
    return 60


def should_wandb_autosync():
    if get_wandb_mode() == "online":
        return False

    use_autosync = os.environ.get(ENV_VAR_WANDB_AUTOSYNC, None)
    if use_autosync is None:
        get_logger().warning(
            "Wandb autosync not specified. Defaults not syncing. "
            f"To enable autosync, set the {ENV_VAR_WANDB_ENABLED} to true."
        )
        use_autosync = "false"
    return use_autosync.lower() == "true"


def should_use_wandb() -> bool:
    if _CONFIG.should_use_wandb is not None:
        return _CONFIG.should_use_wandb

    status = os.environ.get(ENV_VAR_WANDB_ENABLED, None)
    if status is None:
        raise ValueError(
            f"WandB status not set. Define the {ENV_VAR_WANDB_ENABLED} "
            "environment variable as True or False to define whether to use WandB"
        )
    return status.lower() == "true"


def get_wandb_project() -> str:
    if UseWandbProject.global_project is not None:
        return UseWandbProject.global_project

    project = os.environ.get(ENV_VAR_WANDB_PROJECT, None)
    if project is None:
        raise ValueError(
            f"WandB project not set. Define the {ENV_VAR_WANDB_PROJECT} "
            "environment variable to define the project"
        )
    return str(project)


def get_wandb_entity() -> str:
    entity = os.environ.get(ENV_VAR_WANDB_ENTITY, None)
    if entity is None:
        raise ValueError(
            f"WandB entity not set. Define the {ENV_VAR_WANDB_ENTITY} "
            "environment variable to define the entity"
        )
    return str(entity)


def get_wandb_mode() -> str:
    mode = os.environ.get(ENV_VAR_WANDB_MODE, "offline")

    if mode not in ["online", "offline"]:
        raise ValueError(
            f"Invalid wandb mode set in environment variable {ENV_VAR_WANDB_MODE}."
            f"Expected 'online' or 'offline'. Got {mode}."
        )

    return mode


def get_workspace_directory() -> Path:
    workspace = os.environ.get(ENV_VAR_WORKSPACE, None)
    if workspace is None:
        raise ValueError(
            "Workspace not set. "
            f"Define the {ENV_VAR_WORKSPACE} environment variable"
            "To define where to save datasets and experiments results."
        )
    return Path(workspace)


def get_device() -> Literal["cpu", "cuda"]:
    device: Literal["cpu", "cuda"] = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        get_logger().warning("GPU not available, running experiments on CPU.")
    return device


def get_dataset_directory() -> Path:
    return get_workspace_directory() / "datasets"


def get_tokenizers_directory() -> Path:
    return get_workspace_directory() / "tokenizers"


def get_experiment_directory() -> Path:
    return get_workspace_directory() / "experiments"


def get_wandb_cache_directory() -> Path:
    return get_workspace_directory() / "wandb_cache"


def get_console_logging_level() -> str:
    return os.environ.get(ENV_VAR_LOGGING, "DEBUG")


def get_slurm_email() -> str:
    """Email to use for slurm notifications, defined in an environment variable.

    Raises:
         ValueError: if the environment variable is not set.
    """
    email = os.environ.get(ENV_VAR_SLURM_EMAIL, None)
    if email is None:
        raise ValueError(
            "Notification email for Slurm not set. "
            f"Define the {ENV_VAR_SLURM_EMAIL} environment variable."
        )
    return email


def get_slurm_account() -> str:
    """Account to use to submit to slurm, defined in an environment variable.

    Raises:
         ValueError: if the environment variable is not set.
    """
    account = os.environ.get(ENV_VAR_SLURM_ACCOUNT, None)
    if account is None:
        raise ValueError(
            "Slurm Account not set. " f"Define the {account} environment variable."
        )
    return account


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
        sh.setLevel(level=get_console_logging_level() if level is None else level)
        formatter = logging.Formatter(LOG_FMT, datefmt="%Y-%m-%d %H:%M:%S")
        sh.setFormatter(formatter)
        logger.addHandler(sh)
        logger.setLevel(level=get_console_logging_level() if level is None else level)
    return logger


def set_logfile(path: Path, name: Optional[str] = None):
    handler = logging.FileHandler(path)
    handler.formatter = logging.Formatter(LOG_FMT)
    get_logger(name=name).addHandler(handler)
    return handler


def remove_loghandler(handler: logging.FileHandler, name: Optional[str] = None):
    get_logger(name=name).removeHandler(handler)


def get_hash_directory(base_directory: Path, object_hash: str, unique_id: str) -> Path:
    """Get a directory for a unique_id in a hash directory.

    Behaves like a dictionary of paths, returns a unique path for unique_id,
    but uses a hash directory structure to avoid having too many files in a single directory.

    This function manages directory structures that look like as follows::

        base_dir/
        ├─ hash1/
        │  ├─ mapping.json
        │  ├─ 0/
        │  ├─ 1/
        ├─ hash2/
        │  ├─ mapping.json
        │  ├─ 0/
        ├─ hash3/
        ...

    The mapping.json contains a dictionary mapping unique_id to the subdirectory.

    Args:
        base_directory (Path): the base directory containing the hash directories.
        object_hash (str): the hash of the object that the unique_id is associated with
        unique_id (str): the unique identifier for the object
    """

    hash_basedir = base_directory / object_hash
    if not hash_basedir.exists():
        hash_basedir.mkdir()

    mapping_file = hash_basedir / "mapping.json"
    if not mapping_file.exists():
        mapping_file.touch()
        with mapping_file.open("w") as f:
            json.dump({}, f)

    mapping = json.loads(mapping_file.read_text())
    if unique_id in mapping:
        hash_dir = hash_basedir / mapping[unique_id]
    else:
        new_id = len(mapping)
        mapping[unique_id] = str(new_id)
        with mapping_file.open("w") as f:
            json.dump(mapping, f)
        hash_dir = hash_basedir / str(new_id)

    if not hash_dir.exists():
        hash_dir.mkdir()
    return hash_dir
