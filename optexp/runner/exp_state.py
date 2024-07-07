from dataclasses import dataclass
from typing import Iterable

import torch.optim
from torch.utils.data import DataLoader

from optexp.hardwareconfig.hardwareconfig import BatchSizeInfo
from optexp.metrics import Metric


@dataclass
class IterationCounter:
    epoch: int = 0
    step: int = 0
    step_within_epoch: int = 0

    def next_iter(self):
        self.step += 1
        self.step_within_epoch += 1

    def next_epoch(self):
        self.epoch += 1
        self.step_within_epoch = 0


@dataclass
class DataLoaders:
    tr_tr: DataLoader
    tr_va: DataLoader
    va_va: DataLoader


@dataclass
class ExperimentState:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    loss_func: torch.nn.Module
    metrics: Iterable[Metric]
    dataloaders: DataLoaders
    batch_size_info: BatchSizeInfo
    _current_training_dataloader = None
    iteration_counter: IterationCounter = IterationCounter()

    def get_batch(self):
        if self._current_training_dataloader is None:
            self._current_training_dataloader = iter(self.dataloaders.tr_tr)
            self.iteration_counter = IterationCounter()

        try:
            features, labels = next(self._current_training_dataloader)
        except StopIteration:
            self.iteration_counter.next_epoch()
            tr_iterator = iter(self.dataloaders.tr_tr)
            features, labels = next(tr_iterator)
        finally:
            self.iteration_counter.next_iter()

        return features, labels
