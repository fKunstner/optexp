from dataclasses import dataclass

import torch

from optexp.components.optimizers.optimizer import Optimizer, WeightDecayStrategy
from optexp.components.optimizers.weight_decay_strategy import DecayEverything


@dataclass(frozen=True)
class SGD(Optimizer):

    lr: float
    momentum: float = 0
    dampening: float = 0
    weight_decay: float = 0
    nesterov: bool = False
    decay_strategy: WeightDecayStrategy = DecayEverything()

    def load(self, model: torch.nn.Module) -> torch.optim.Optimizer:

        param_groups = self.decay_strategy.make_param_groups(model, self.weight_decay)

        return torch.optim.SGD(
            param_groups,
            lr=self.lr,
            momentum=self.momentum,
            dampening=self.dampening,
            weight_decay=self.weight_decay,
            nesterov=self.nesterov,
        )
