import torch
from attrs import frozen

from optexp.optim.optimizer import Optimizer, Regularizable, WeightDecayStrategy
from optexp.optim.weight_decay_strategies import DecayEverything


@frozen
class Adam(Optimizer, Regularizable):
    """Adam optimizer from [Kingma2014]_.

    Args:
        lr (float): learning rate.
        beta1 (float, optional): coefficient used for computing EMA of gradient.
            Defaults to 0.9.
        beta2 (float, optional): coefficient used for computing EMA of squared gradients.
            Defaults to 0.999.
        eps (float, optional): term added to the denominator to improve numerical stability.
            Defaults to 1e-8.
        weight_decay (float, optional): weight decay (L2 penalty). Defaults to 0.01.
        amsgrad (bool, optional): whether to use the AMSGrad variant of this algorithm.
            Defaults to False.
        decay_strategy (WeightDecayStrategy, optional): strategy for applying weight decay.
            Defaults to :class:`~optexp.optim.DecayEverything()`.

    .. [Kingma2014] Adam: A Method for Stochastic Optimization.
       Diederik P. Kingma, Jimmy Ba.
       International Conference on Learning Representations, 2015.
       `doi.org/10.48550/arXiv.1412.6980 <https://doi.org/10.48550/arXiv.1412.6980>`_
    """

    lr: float
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.01
    amsgrad: bool = False
    decay_strategy: WeightDecayStrategy = DecayEverything()

    def regularizer_loss(self, model: torch.nn.Module) -> torch.Tensor:
        return self.decay_strategy.regularizer_loss(model, self.weight_decay)

    def load(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        param_groups = self.decay_strategy.make_param_groups(model, self.weight_decay)
        return torch.optim.Adam(
            param_groups,
            lr=self.lr,
            betas=(self.beta1, self.beta2),
            eps=self.eps,
            weight_decay=self.weight_decay,
            amsgrad=self.amsgrad,
        )

    def plot_name(self) -> str:
        attributes = []
        if self.lr is not None and self.lr != 0:
            attributes.append(rf"$\alpha={self.lr:.3g}$")
        if self.beta1 is not None:
            attributes.append(rf"$\beta_1={self.beta1:.3g}$")
        if self.beta2 is not None:
            attributes.append(rf"$\beta_2={self.beta2:.3g}$")
        if self.weight_decay is not None and self.weight_decay != 0:
            attributes.append(rf"$\lambda={self.weight_decay:.3g}$")
        return self.__class__.__name__ + " (" + " ".join(attributes) + ")"


@frozen
class AdamW(Adam, Regularizable):
    """AdamW optimizer from [Loshchilov2019]_.

    Args:
        lr (float): learning rate.
        beta1 (float, optional): coefficient used for computing EMA of gradient.
            Defaults to 0.9.
        beta2 (float, optional): coefficient used for computing EMA squared gradient.
            Defaults to 0.999.
        eps (float, optional): term added to the denominator to improve numerical stability.
            Defaults to 1e-8.
        weight_decay (float, optional): weight decay (L2 penalty). Defaults to 0.01.
        amsgrad (bool, optional): whether to use the AMSGrad variant of this algorithm.
            Defaults to False.
        decay_strategy (WeightDecayStrategy, optional): strategy for applying weight decay.
            Defaults to :class:`~optexp.optim.DecayEverything()`.

    .. [Loshchilov2019] Decoupled Weight Decay Regularization.
         Ilya Loshchilov, Frank Hutter.
         International Conference on Learning Representations, 2019.
         `doi.org/10.48550/arXiv.1711.05101 <https://doi.org/10.48550/arXiv.1711.05101>`_
    """

    def regularizer_loss(self, model: torch.nn.Module) -> torch.Tensor:
        return self.decay_strategy.regularizer_loss(model, self.weight_decay)

    def load(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        param_groups = self.decay_strategy.make_param_groups(model, self.weight_decay)
        return torch.optim.AdamW(
            param_groups,
            lr=self.lr,
            betas=(self.beta1, self.beta2),
            eps=self.eps,
            weight_decay=self.weight_decay,
            amsgrad=self.amsgrad,
        )
