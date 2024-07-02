from dataclasses import dataclass

import torch

from optexp.optimizers.optimizer import Optimizer
from optexp.optimizers.weight_decay_strategy import DecayEverything, WeightDecayStrategy


@dataclass(frozen=True)
class Adagrad(Optimizer):

    lr: float
    weight_decay: float = 0.0
    lr_decay: float = 0.0
    decay_strategy: WeightDecayStrategy = DecayEverything()

    def load(self, model: torch.nn.Module) -> torch.optim.Optimizer:

        param_groups = self.decay_strategy.make_param_groups(model, self.weight_decay)

        return torch.optim.Adagrad(
            param_groups,
            lr=self.lr,
            lr_decay=self.lr_decay,
            weight_decay=self.weight_decay,
        )
