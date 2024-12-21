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


class TransformerInitialization(InitializationStrategy):

    def initialize(self, model: torch.nn.Module) -> torch.nn.Module:
        raise NotImplementedError  # TODO
