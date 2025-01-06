from dataclasses import dataclass, field
from typing import Optional

import lightning
import torch.optim
from torch.utils.data import DataLoader

from optexp.hardwareconfig.hardwareconfig import BatchSizeInfo


@dataclass
class IterationCounter:
    epoch: int = 0
    step: int = 0
    step_within_epoch: int = 0

    def start(self):
        self.step = 0
        self.step_within_epoch = 0
        self.epoch = 1

    def next_iter(self):
        self.step += 1
        self.step_within_epoch += 1

    def next_epoch(self):
        self.epoch += 1
        self.step_within_epoch = 0


@dataclass
class DataLoaders:
    tr_tr: DataLoader
    eval_tr: DataLoader
    eval_va: DataLoader
    eval_te: Optional[DataLoader] = None

    def get_val_dataloader(self, split):
        if split == "tr":
            return self.eval_tr
        if split == "va":
            return self.eval_va
        if split == "te":
            return self.eval_te
        raise ValueError(f"Unknown split: {split}")


@dataclass
class ExperimentState:
    # pylint: disable=too-many-instance-attributes
    model: lightning.fabric.wrappers._FabricModule  # pylint: disable=protected-access
    optimizer: torch.optim.Optimizer
    dataloaders: DataLoaders
    batch_size_info: BatchSizeInfo
    _current_training_dataloader = None
    iteration_counter: IterationCounter = field(default_factory=IterationCounter)

    # TODO is this the best way to fix gradient accumulation?
    def get_batch(self, is_accumulating: bool = False):
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
            if not is_accumulating:
                self.iteration_counter.next_iter()

        return data
