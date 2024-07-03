from abc import ABC, abstractmethod

import torch

from optexp.components.component import Component, dataclass_component


@dataclass_component()
class Model(Component, ABC):

    @abstractmethod
    def load_model(
        self, input_shape: torch.Size, output_shape: torch.Size
    ) -> torch.nn.Module:
        raise NotImplementedError()
