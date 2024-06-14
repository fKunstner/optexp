from typing import Tuple

import torch
from torch.nn.functional import cross_entropy, mse_loss


class Accuracy(torch.nn.Module):
    def __init__(self, reduction="mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, inputs, labels):
        classes = torch.argmax(inputs, dim=1)
        if self.reduction == "mean":
            return torch.mean((classes == labels).float())
        return torch.sum((classes == labels).float())


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


class CrossEntropyLossPerClass(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def forward(inputs, labels):
        num_classes = inputs.shape[1]
        losses = cross_entropy(inputs, labels, reduction="none")
        return _groupby_sum(losses, labels, num_classes)


class MSELossPerClass(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def forward(inputs, labels):
        num_classes = inputs.shape[1]
        # pylint: disable=not-callable
        one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=num_classes)
        losses = mse_loss(inputs, one_hot_labels, reduction="none")
        return _groupby_sum(losses.mean(axis=1), labels, num_classes)


class AccuracyPerClass(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def forward(inputs, labels):
        num_classes = inputs.shape[1]
        classes = torch.argmax(inputs, dim=1)
        accuracy_per_sample = (classes == labels).float()
        return _groupby_sum(accuracy_per_sample, labels, num_classes)


class ClassificationSquaredLoss(torch.nn.Module):
    def __init__(self, reduction="mean") -> None:
        super().__init__()
        self.reduction = reduction

    @staticmethod
    def forward(inputs, labels):
        num_classes = inputs.shape[1]
        # pylint: disable=not-callable
        one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=num_classes)
        class_sum = torch.sum(
            (torch.masked_select(inputs, one_hot_labels > 0) - 1) ** 2
        )
        output = (1.0 / num_classes) * (
            class_sum
            + torch.sum(torch.square(torch.masked_select(inputs, one_hot_labels == 0)))
        )
        return output
