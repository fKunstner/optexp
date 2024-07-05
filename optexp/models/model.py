from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from optexp.component import Component


@dataclass(frozen=True)
class Model(Component, ABC):
    """Abstract base class for models."""

    @abstractmethod
    def load_model(
        self, input_shape: torch.Size, output_shape: torch.Size
    ) -> torch.nn.Module:
        raise NotImplementedError()
