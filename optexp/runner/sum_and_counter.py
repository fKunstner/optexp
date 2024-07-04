from dataclasses import dataclass
from typing import Tuple

import lightning as ptl
import torch
from torch import Tensor

from optexp.runner.fabric_helpers import reduce_tensor


@dataclass
class SumAndCounter:
    numerator: Tensor
    denominator: Tensor

    def __add__(self, other: "SumAndCounter") -> "SumAndCounter":
        return SumAndCounter(
            self.numerator + other.numerator, self.denominator + other.denominator
        )

    def reduce(self, fabric: ptl.Fabric, reduce_op="sum") -> Tensor:
        numerator = reduce_tensor(fabric, self.numerator, reduce_op=reduce_op)
        denominator = reduce_tensor(fabric, self.denominator, reduce_op=reduce_op)
        return numerator / denominator

    def result(self) -> Tensor:
        return self.numerator / self.denominator

    @staticmethod
    def from_output(output: Tensor | Tuple[Tensor, Tensor]):
        if isinstance(output, tuple) and len(output) == 2:
            return SumAndCounter(output[0], output[1])
        return SumAndCounter(output, torch.tensor(1))
