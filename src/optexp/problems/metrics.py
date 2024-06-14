import torch
from torch import nn as nn
from torch.nn.functional import cross_entropy, mse_loss


class Accuracy(nn.Module):
    def __init__(self, reduction="mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, inputs, labels):
        classes = torch.argmax(inputs, dim=1)
        if self.reduction == "mean":
            return torch.mean((classes == labels).float())
        else:
            return torch.sum((classes == labels).float())


def _groupby_average(inputs: torch.Tensor, classes, num_classes):
    """Given inputs and classes, both of size [n], where classes contains indices in [1,
    ..., num_classes]

    - `sum_by_class`: [num_classes] containing the sum of the inputs per class
    - `label_counts`: [num_classes] containing the number of elements per class
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


class CrossEntropyLossPerClass(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def forward(inputs, labels):
        num_classes = inputs.shape[1]
        losses = cross_entropy(inputs, labels, reduction="none")
        return _groupby_average(losses, labels, num_classes)


class MSELossPerClass(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def forward(inputs, labels):
        num_classes = inputs.shape[1]
        one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=num_classes)
        losses = mse_loss(inputs, one_hot_labels, reduction="none")
        return _groupby_average(losses.mean(axis=1), labels, num_classes)


class AccuracyPerClass(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def forward(inputs, labels):
        num_classes = inputs.shape[1]
        classes = torch.argmax(inputs, dim=1)
        accuracy_per_sample = (classes == labels).float()
        return _groupby_average(accuracy_per_sample, labels, num_classes)


class ClassificationSquaredLoss(nn.Module):
    def __init__(self, reduction="mean") -> None:
        super().__init__()
        self.reduction = reduction

    @staticmethod
    def forward(inputs, labels):
        num_classes = inputs.shape[1]
        one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=num_classes)
        class_sum = torch.sum(
            (torch.masked_select(inputs, one_hot_labels > 0) - 1) ** 2
        )
        output = (1.0 / num_classes) * (
            class_sum
            + torch.sum(torch.square(torch.masked_select(inputs, one_hot_labels == 0)))
        )
        return output
