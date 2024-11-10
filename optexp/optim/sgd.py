import torch
from attrs import frozen

from optexp.optim.optimizer import Optimizer, Regularizable, WeightDecayStrategy
from optexp.optim.weight_decay_strategies import DecayEverything


@frozen
class SGD(Optimizer, Regularizable):
    """Stochastic Gradient Descent.

    Args:
        lr (float): learning rate.
        momentum (float, optional): momentum. Defaults to 0
        dampening (float, optional): dampening for momentum. Defaults to 0
        weight_decay (float, optional): weight decay (L2 penalty). Defaults to 0
        nesterov (bool, optional): enables Nesterov momentum. Defaults to False
        decay_strategy (WeightDecayStrategy, optional): The strategy for applying weight decay.
           Defaults to DecayEverything().
    """

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

    def regularizer_loss(self, model: torch.nn.Module) -> torch.Tensor:
        return self.decay_strategy.regularizer_loss(model, self.weight_decay)

    def plot_name(self) -> str:
        attributes = []
        if self.lr is not None and self.lr != 0:
            attributes.append(rf"$\alpha={self.lr:.3g}$")
        if self.momentum is not None and self.momentum != 0:
            attributes.append(rf"$\beta={self.momentum:.3g}")
        if self.weight_decay is not None and self.weight_decay != 0:
            attributes.append(rf"$\lambda={self.weight_decay:.3g}")
        return "GD (" + " ".join(attributes) + ")"
