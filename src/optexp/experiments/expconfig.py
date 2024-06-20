import dataclasses
from abc import ABC, abstractmethod
from typing import Literal, Optional

import torch
from lightning import Fabric

from optexp import config
from optexp.datasets import Dataset
from optexp.datasets.dataset import TrVa


class ExpConfig(ABC):

    @abstractmethod
    def load(self, fabric: Fabric, dataset: Dataset) -> "DetailedExpConfig":
        raise NotImplementedError

    @abstractmethod
    def get_num_workers(self):
        raise NotImplementedError

    @abstractmethod
    def get_accelerator(self) -> Literal["cpu", "cuda"]:
        raise NotImplementedError

    @abstractmethod
    def get_seed(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def use_wandb_autosync(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_eval_every(self) -> int:
        raise NotImplementedError


class RawExpConfig(ExpConfig):

    def __init__(
        self,
        seed: int,
        batch_size: int,
        eval_every: int,
        micro_batch_size: int,
        num_workers: int,
        eval_micro_batch_size: Optional[int] = None,
        device: Literal["cpu", "cuda", "auto"] = "auto",
        epochs: Optional[int] = None,
        steps: Optional[int] = None,
        wandb_autosync: bool = False,
    ):
        if epochs is not None and steps is not None:
            raise ValueError("Cannot have both epochs and steps")
        if epochs is None and steps is None:
            raise ValueError("Must have either epochs or steps")
        if steps is not None and steps < 1:
            raise ValueError("Steps must be greater than 0")
        if epochs is not None and epochs < 1:
            raise ValueError("Epochs must be greater than 0")
        if batch_size % (micro_batch_size * num_workers) != 0:
            raise ValueError(
                "Batch size must be a multiple of micro batch size * num workers"
            )

        self.seed = seed
        self.batch_size = batch_size
        self.eval_every = eval_every
        self.micro_batch_size = micro_batch_size
        self.num_workers = num_workers
        self.eval_micro_batch_size = eval_micro_batch_size
        self.device = device
        self.epochs = epochs
        self.steps = steps
        self.wandb_autosync = wandb_autosync

    def load(self, fabric: Fabric, dataset: Dataset) -> "RawDetailedExpConfig":
        return RawDetailedExpConfig(self, fabric, dataset)

    def get_num_workers(self):
        return self.num_workers

    def get_accelerator(self) -> Literal["cpu", "cuda"]:
        match self.device:
            case "auto":
                return config.get_device()
            case "cpu":
                return "cpu"
            case "cuda":
                return "cuda"
            case _:
                raise ValueError(f"Unknown device {self.device}")

    def get_seed(self) -> int:
        return self.seed

    def use_wandb_autosync(self) -> bool:
        return self.wandb_autosync

    def get_eval_every(self) -> int:
        return self.eval_every


class DetailedExpConfig(ABC):

    @abstractmethod
    def get_batch_size_for_dataloader(self, used_for_tr_va: TrVa) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_steps(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_gradient_accumulation_steps(self) -> int:
        raise NotImplementedError


class RawDetailedExpConfig(DetailedExpConfig):

    def __init__(
        self,
        exp_config: RawExpConfig,
        fabric: Fabric,
        dataset: Dataset,
    ):
        self._steps: int = self._figure_out_steps(exp_config, dataset)
        tr_b, tr_microb, va_microb = self._validate_batch_sizes(exp_config, dataset)
        self.tr_b = tr_b
        self.tr_microb = tr_microb
        self.va_microb = va_microb
        self.exp_config: RawExpConfig = exp_config

    def get_steps(self) -> int:
        return self._steps

    def get_batch_size_for_dataloader(self, tr_va_dataloader: TrVa) -> int:
        if tr_va_dataloader == "tr":
            return self.tr_microb
        if tr_va_dataloader == "va":
            return self.va_microb
        raise ValueError(f"Unknown tr_va_dataloader {tr_va_dataloader}")

    @staticmethod
    def _figure_out_steps(exp_config: RawExpConfig, dataset: Dataset) -> int:
        n_tr = dataset.get_num_samples("tr")
        if exp_config.steps is None:
            steps_per_epoch = n_tr // exp_config.batch_size

            if exp_config.epochs is None:
                raise ValueError("Either epochs or steps must be set")

            return exp_config.epochs * steps_per_epoch
        return exp_config.steps

    @staticmethod
    def _validate_batch_sizes(exp_config, dataset):
        n_tr = dataset.get_num_samples("tr")
        n_va = dataset.get_num_samples("va")

        if n_tr % exp_config.batch_size != 0:
            raise ValueError(
                "Error in the batch size for training dataloader."
                "Batch size must divide number of training samples."
                f"Got batch size: {exp_config.batch_size},"
                f"number of training samples: {n_tr}"
            )

        if (
            exp_config.batch_size
            % (exp_config.micro_batch_size * exp_config.num_workers)
            != 0
        ):
            raise ValueError(
                "Batch size must be a multiple of micro batch size * num workers"
            )

        val_microb = (
            exp_config.eval_micro_batch_size
            if exp_config.eval_micro_batch_size is not None
            else exp_config.micro_batch_size
        )
        if n_va % val_microb != 0:
            raise ValueError(
                "Error in the micro batch size for evaluation dataloader."
                "Micro batch size must divide number of validation samples."
                f"Got micro batch size: {val_microb},"
                f"number of validation samples: {n_va}"
            )

        return exp_config.batch_size, exp_config.micro_batch_size, val_microb

    def get_gradient_accumulation_steps(self) -> int:
        return self.tr_b // self.tr_microb
