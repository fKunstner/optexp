from dataclasses import dataclass

from torch.nn import Module
from torch.optim import SGD as TorchSGD
from torch.optim import Optimizer as TorchOptimizer

from optexp.optimizers.hyperparameter import LearningRate
from optexp.optimizers.optimizer import Optimizer
from optexp.optimizers.weight_decay_strategy import DecayEverything, WeightDecayStrategy


@dataclass
class SGD(Optimizer):

    lr: LearningRate
    momentum: float = 0
    dampening: float = 0
    weight_decay: float = 0
    nesterov: bool = False
    decay_strategy: WeightDecayStrategy = DecayEverything()

    def load(self, model: Module) -> TorchOptimizer:

        param_groups = self.decay_strategy.make_param_groups(model, self.weight_decay)

        return TorchSGD(
            param_groups,
            lr=self.lr.as_float(),
            momentum=self.momentum,
            dampening=self.dampening,
            weight_decay=self.weight_decay,
            nesterov=self.nesterov,
        )
