from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass()
class Model(ABC):

    @abstractmethod
    def load_model(
        self, input_shape: torch.Size, output_shape: torch.Size
    ) -> torch.nn.Module:
        raise NotImplementedError()
