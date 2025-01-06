from typing import Dict, Iterable, List, Union

from torch.nn import Module, Parameter

from optexp.optim.optimizer import WeightDecayStrategy


class DecayEverything(WeightDecayStrategy):
    """Applies weight decay to all parameters."""

    def make_param_groups(
        self, model: Module, weight_decay: float
    ) -> List[Dict[str, Union[Iterable[Parameter], float]]]:
        return [{"params": model.parameters(), "weight_decay": weight_decay}]


class NoDecayOnBias(WeightDecayStrategy):
    """Applies weight decay to all parameters except biases.

    Only applies weight decay to parameters whose name does not contain "bias".
    """

    def make_param_groups(
        self, model: Module, weight_decay: float
    ) -> List[Dict[str, Union[Iterable[Parameter], float]]]:
        return [
            {
                "params": (p for n, p in model.named_parameters() if "bias" not in n),
                "weight_decay": weight_decay,
            },
            {
                "params": (p for n, p in model.named_parameters() if "bias" in n),
                "weight_decay": 0.0,
            },
        ]


class GPT2WeightDecay(WeightDecayStrategy):
    """
    Applies on weight decay on  matrices but not vectors.

    Bias vectors and  Normalization Layers do not have weight decay but linear layers  do.
    """

    def make_param_groups(
        self, model: Module, weight_decay: float
    ) -> List[Dict[str, Union[Iterable[Parameter], float]]]:
        return [
            {
                "params": (
                    p
                    for n, p in model.named_parameters()
                    if p.dim() >= 2 and p.requires_grad
                ),
                "weight_decay": weight_decay,
            },
            {
                "params": (
                    p
                    for n, p in model.named_parameters()
                    if p.dim() < 2 and p.requires_grad
                ),
                "weight_decay": 0.0,
            },
        ]
