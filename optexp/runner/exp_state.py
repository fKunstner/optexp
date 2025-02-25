from dataclasses import dataclass, field
from typing import Optional

import lightning
import torch.optim
from torch.utils.data import DataLoader

from optexp.hardwareconfig.hardwareconfig import BatchSizeInfo


@dataclass
class IterationCounter:
    epoch: int = 0
    microbatches: int = 0
    microbatches_within_epoch: int = 0
    steps: int = 0
    steps_within_epoch: int = 0

    def start(self):
        self.microbatches = 0
        self.microbatches_within_epoch = 0
        self.steps = 0
        self.steps_within_epoch = 0
        self.epoch = 1

    def next_microbatch(self):
        self.microbatches += 1
        self.microbatches_within_epoch += 1

    def step(self):
        self.steps += 1
        self.steps_within_epoch += 1

    def next_epoch(self):
        self.epoch += 1
        self.microbatches_within_epoch = 0
        self.steps_within_epoch = 0


@dataclass
class DataLoaders:
    tr_tr: DataLoader
    eval_tr: Optional[DataLoader] = None
    eval_va: Optional[DataLoader] = None
    eval_te: Optional[DataLoader] = None

    def get_val_dataloader(self, split) -> DataLoader:
        dataloader: Optional[DataLoader] = None
        if split == "tr":
            dataloader = self.eval_tr
        elif split == "va":
            dataloader = self.eval_va
        elif split == "te":
            dataloader = self.eval_te

        if dataloader is None:
            raise ValueError(
                f"Dataloader for split {split} was requested, but it was not initialized."
            )

        return dataloader


@dataclass
class ExperimentState:
    # pylint: disable=too-many-instance-attributes
    model: lightning.fabric.wrappers._FabricModule  # pylint: disable=protected-access
    optimizer: torch.optim.Optimizer
    dataloaders: DataLoaders
    batch_size_info: BatchSizeInfo
    _current_training_dataloader = None
    iteration_counter: IterationCounter = field(default_factory=IterationCounter)

    def get_microbatch(self):
        if self._current_training_dataloader is None:
            self._current_training_dataloader = iter(self.dataloaders.tr_tr)
            self.iteration_counter.start()

        try:
            data = next(self._current_training_dataloader)
        except StopIteration:
            self.iteration_counter.next_epoch()
            self._current_training_dataloader = iter(self.dataloaders.tr_tr)
            data = next(self._current_training_dataloader)
        finally:
            self.iteration_counter.next_microbatch()

        return data

    def step(self):
        self.iteration_counter.step()
