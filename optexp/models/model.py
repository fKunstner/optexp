from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from optexp.component import Component


def assert_batch_sizes_match(b1: int, b2: int) -> None:
    assert b1 == b2, f"Batch sizes do not match: {b1} != {b2}"


@dataclass(frozen=True)
class Model(Component, ABC):
    """Abstract base class for models."""

    @abstractmethod
    def load_model(
        self, input_shape: torch.Size, output_shape: torch.Size
    ) -> torch.nn.Module:
        raise NotImplementedError()
