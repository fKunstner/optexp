from typing import List

import torch

from optexp.problems.problem import Problem
from optexp.problems.utils import Accuracy, AccuracyPerClass, CrossEntropyLossPerClass


class Classification(Problem):
    def init_loss(self) -> torch.nn.Module:
        return torch.nn.CrossEntropyLoss()

    def get_criterions(self) -> List[torch.nn.Module]:
        return [torch.nn.CrossEntropyLoss(), Accuracy()]


class ClassificationWithPerClassStats(Classification):
    def init_loss(self) -> torch.nn.Module:
        return torch.nn.CrossEntropyLoss()

    def get_criterions(self) -> List[torch.nn.Module]:
        return [
            torch.nn.CrossEntropyLoss(),
            Accuracy(),
            CrossEntropyLossPerClass(),
            AccuracyPerClass(),
        ]
