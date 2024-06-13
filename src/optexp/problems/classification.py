from typing import List

import torch

from optexp.problems.metrics import Accuracy
from optexp.problems.problem import Problem


class Classification(Problem):
    def init_loss(self) -> torch.nn.Module:
        return torch.nn.CrossEntropyLoss()

    def get_criterions(self) -> List[torch.nn.Module]:
        return [torch.nn.CrossEntropyLoss(), Accuracy()]
