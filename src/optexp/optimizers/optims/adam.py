from dataclasses import dataclass

from torch.nn import Module
from torch.optim import Adam as TorchAdam
from torch.optim import Optimizer as TorchOptimizer

from optexp.optimizers.hyperparameter import LearningRate
from optexp.optimizers.optimizer import Optimizer
from optexp.optimizers.weight_decay_strategy import DecayEverything, WeightDecayStrategy


@dataclass
class Adam(Optimizer):
    lr: LearningRate
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.01
    amsgrad: bool = False
    decay_strategy: WeightDecayStrategy = DecayEverything()

    def load(self, model: Module) -> TorchOptimizer:
        param_groups = self.decay_strategy.make_param_groups(model, self.weight_decay)
        return TorchAdam(
            param_groups,
            lr=self.lr.as_float(),
            betas=(self.beta1, self.beta2),
            eps=self.eps,
            weight_decay=self.weight_decay,
            amsgrad=self.amsgrad,
        )
