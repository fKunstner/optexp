import math
from abc import ABC, abstractmethod

import torch.nn

from optexp.component import Component


class InitializationStrategy(ABC, Component):
    """Abstract base class for parameter initialization"""

    @abstractmethod
    def initialize(self, model: torch.nn.Module) -> torch.nn.Module:
        raise NotImplementedError()


class DefaultInitialization(InitializationStrategy):
    def initialize(self, model: torch.nn.Module) -> torch.nn.Module:
        return model


# mostly stolen from https://github.com/karpathy/nanoGPT/blob/master/model.py
class GPT2Initialization(InitializationStrategy):

    def initialize(self, model: torch.nn.Module) -> torch.nn.Module:
        if not hasattr(model, "encoder"):
            raise ValueError("GPT2 Initialization requires model to have an encoder!")
        n_layers = getattr(model.encoder, "num_layers")

        def init_weights(module: torch.nn.Module):
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, torch.nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        model.apply(init_weights)

        for pn, p in model.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))
        return model
