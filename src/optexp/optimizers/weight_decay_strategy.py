from dataclasses import dataclass
from typing import Dict, Iterable, List, Union

from torch.nn import Module, Parameter

from optexp.experiments.component import Component


@dataclass(frozen=True)
class WeightDecayStrategy(Component):

    def make_param_groups(
        self, model: Module, weight_decay: float
    ) -> List[Dict[str, Union[Iterable[Parameter], float]]]:
        raise NotImplementedError


class DecayEverything(WeightDecayStrategy):
    def make_param_groups(
        self, model: Module, weight_decay: float
    ) -> List[Dict[str, Union[Iterable[Parameter], float]]]:
        return [{"params": model.parameters(), "weight_decay": weight_decay}]


class NoDecayOnBias(WeightDecayStrategy):
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
