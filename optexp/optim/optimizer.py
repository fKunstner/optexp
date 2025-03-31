from abc import ABC, abstractmethod
from typing import (
    Dict,
    Iterable,
    List,
    Union,
    Optional,
    Callable,
    overload,
    Any,
    Iterator,
    Tuple,
)

import torch
from attrs import frozen
from torch.nn import Module, Parameter

from optexp.component import Component


@frozen
class Optimizer(Component, ABC):
    """Abstract base class for optimizers."""

    @abstractmethod
    def load(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        pass

    def plot_style(self):
        return {}


@frozen
class OptimGroups(Component):
    embeddings: Optimizer
    default: Optimizer
    prediction_layer: Optimizer


@frozen
class MultiOptimizer(Optimizer):
    """A wrapper for multiple optimizers"""

    optimizer_groups: OptimGroups

    def load(self, model: torch.nn.Module) -> torch.optim.Optimizer:

        optimizer_groups = {
            "embeddings": self.optimizer_groups.embeddings,
            "default": self.optimizer_groups.default,
            "prediction_layer": self.optimizer_groups.prediction_layer,
        }

        if "default" not in optimizer_groups.keys():
            raise ValueError(
                "Please specify a default optimizer for parameters not specified"
            )

        parameter_groups = {m: {} for m in optimizer_groups.keys()}
        named_modules: Dict[str : torch.nn.Module] = dict(model.named_modules())
        skip_modules = list(optimizer_groups.keys()).copy()

        for m in named_modules.keys():
            if m in optimizer_groups.keys():
                named_params = dict(named_modules[m].named_parameters(recurse=True))
                named_params = {f"{m}.{k}": v for k, v in named_params.items()}
                parameter_groups[m] = {**parameter_groups[m], **named_params}
                skip_modules += [
                    f"{m}.{c}" for c in dict(named_modules[m].named_children()).keys()
                ]
        for m in named_modules.keys():
            if m in skip_modules:
                continue
            named_params = dict(named_modules[m].named_parameters(recurse=False))
            named_params = {f"{m}.{k}": v for k, v in named_params.items()}
            parameter_groups["default"] = {
                **parameter_groups["default"],
                **named_params,
            }
        torch_optimizers = {}
        for key, opt in optimizer_groups.items():
            named_param_dict = parameter_groups[key]

            class DummyModule(torch.nn.Module):
                def named_parameters(
                    self,
                    prefix: str = "",
                    recurse: bool = True,
                    remove_duplicate: bool = True,
                ) -> Iterator[Tuple[str, Parameter]]:
                    yield from named_param_dict.items()

            torch_opt = opt.load(DummyModule())
            torch_optimizers[key] = torch_opt

        total_trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        optimizer_assigned_parameters = 0

        for opt in torch_optimizers.values():
            for group in opt.param_groups:
                for param in group["params"]:
                    optimizer_assigned_parameters += param.numel()

        if total_trainable_params != optimizer_assigned_parameters:
            raise ValueError(
                f"MultiOptimizer is Missing Parameters."
                f"Trainable Parameters:{total_trainable_params}"
                f"Assigned Parameters:{optimizer_assigned_parameters}"
            )

        torch_multioptimizer = TorchMultiOptimizer(
            model.parameters(),
            defaults=optimizer_groups,
            torch_optimizers=list(torch_optimizers.values()),
        )
        return torch_multioptimizer

        # todo some sort of validation that all parameters groups of the model are covered

        #  dict has what goes where, assign all things with  dict  key, if there is a default key assign the rest to that
        # if there is no default and  some parameter is missed,  raise value error
        # also validate that named_parameter  keys contians all  speicifed optimizer groups,

        # init the  torch opts  empty  and add parameter group by  module?

        # assert that children is empty when assigning parameters, then assert the opt state param count is the same as the model param count

        # oirgnjakleoifsngklarerw there are non-leaf parameters... need to detect

        # the optexp load  needs  all the parameters, we cant add param groups after  the fact (will mess up  things like weight decay  strategy)
        # create  dummy module that we  tie the params  we care about to?


class Regularizable(ABC):
    """Abstract base class for regular"""

    @abstractmethod
    def regularizer_loss(self, model: torch.nn.Module) -> torch.Tensor:
        pass


@frozen
class WeightDecayStrategy(Component):
    """Abstract base class for weight decay strategies."""

    def make_param_groups(
        self, model: Module, weight_decay: float
    ) -> List[Dict[str, Union[Iterable[Parameter], float]]]:
        raise NotImplementedError

    def regularizer_loss(self, model: Module, weight_decay: float) -> torch.Tensor:
        loss = None
        for group in self.make_param_groups(model, weight_decay):
            if group["weight_decay"] != 0:  # type: ignore
                for p in group["params"]:  # type: ignore
                    if loss is None:
                        loss = group["weight_decay"] * p.norm(p=2) ** 2
                    else:
                        loss += group["weight_decay"] * p.norm(p=2) ** 2

        if loss is None:
            return torch.tensor(0.0, device=next(model.parameters()).device)
        return loss  # type: ignore


class TorchMultiOptimizer(torch.optim.Optimizer):
    torch_optimizers: List[torch.optim.Optimizer]

    def __init__(
        self,
        params,
        defaults: Dict[str, Any],
        torch_optimizers: List[torch.optim.Optimizer],
    ):
        super().__init__(params, defaults)
        self.torch_optimizers = torch_optimizers

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        for opt in self.torch_optimizers:
            loss = opt.step(closure=closure)
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        for opt in self.torch_optimizers:
            opt.zero_grad(set_to_none=set_to_none)
