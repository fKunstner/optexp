from typing import Tuple

import torch
from torch.nn.functional import cross_entropy, mse_loss

from optexp.metrics.metric import LossLikeMetric, Metric


class MSE(Metric):
    def __call__(
        self, inputs: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return mse_loss(inputs, labels, reduction="sum"), torch.tensor(labels.numel())

    def smaller_better(self) -> bool:
        return True


class CrossEntropy(Metric):
    def __call__(
        self, inputs: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return cross_entropy(inputs, labels, reduction="sum"), torch.tensor(
            labels.numel()
        )

    def smaller_better(self) -> bool:
        return True

    def is_scalar(self):
        return True


class Accuracy(Metric):

    def __call__(
        self, inputs: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        classes = torch.argmax(inputs, dim=1)
        return torch.sum((classes == labels).float()), torch.tensor(classes.numel())

    def smaller_better(self) -> bool:
        return False

    def is_scalar(self):
        return True


def _groupby_sum(
    inputs: torch.Tensor, classes, num_classes
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sums by class.

    Args:
        inputs: Tensor of size [n]
        classes: Tensor of size [n] containing indices in [1, ..., num_classes]
        num_classes: Number of classes

    Returns:
        tuple of (sum_by_class, label_counts) where
        sum_by_class: [num_classes] containing the sum of the inputs per class
        label_counts: [num_classes] containing the number of elements per class

    such that
        sum_by_class[c] == sum(inputs[classes == c])
        label_counts[c] == sum(classes == c)
    """
    classes = classes.view(-1)

    label_counts = torch.zeros(num_classes, dtype=torch.float, device=classes.device)
    label_counts = label_counts.scatter_add_(0, classes, torch.ones_like(inputs))

    sum_by_class = torch.zeros(num_classes, dtype=torch.float, device=classes.device)
    sum_by_class = sum_by_class.scatter_add_(0, classes, inputs)

    return sum_by_class, label_counts


class CrossEntropyPerClass(LossLikeMetric):
    """Cross entropy loss per class.

    Can result in large logs on problems with many classes.
    """

    def __call__(self, inputs, labels):
        num_classes = inputs.shape[1]
        losses = cross_entropy(inputs, labels, reduction="none")
        return _groupby_sum(losses, labels, num_classes)

    def smaller_better(self) -> bool:
        return True

    def is_scalar(self):
        return False


class AccuracyPerClass(LossLikeMetric):
    """Accuracy per class.

    Can result in large logs on problems with many classes.
    """

    def __call__(self, inputs, labels):
        num_classes = inputs.shape[1]
        classes = torch.argmax(inputs, dim=1)
        accuracy_per_sample = (classes == labels).float()
        return _groupby_sum(accuracy_per_sample, labels, num_classes)

    def smaller_better(self) -> bool:
        return True

    def is_scalar(self):
        return False
