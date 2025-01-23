import warnings
from typing import Tuple

import torch
from attr import frozen
from torch import Tensor
from torch.nn.functional import cross_entropy, l1_loss, mse_loss

from optexp.datasets.dataset import HasClassCounts
from optexp.datastructures import ExpInfo
from optexp.metrics.metric import LossLikeMetric


class MSE(LossLikeMetric):

    def smaller_is_better(self) -> bool:
        return True

    def is_scalar(self) -> bool:
        return True

    def __call__(
        self, inputs: Tensor, labels: Tensor, exp_info: ExpInfo
    ) -> Tuple[Tensor, Tensor]:
        return mse_loss(inputs, labels, reduction="sum"), torch.tensor(labels.numel())

    def unreduced_call(
        self, inputs: Tensor, labels: Tensor, exp_info: ExpInfo
    ) -> Tensor:
        return mse_loss(inputs, labels, reduction="none")


class MAE(LossLikeMetric):

    def smaller_is_better(self) -> bool:
        return True

    def is_scalar(self) -> bool:
        return True

    def __call__(
        self, inputs: Tensor, labels: Tensor, exp_info: ExpInfo
    ) -> Tuple[Tensor, Tensor]:
        return l1_loss(inputs, labels, reduction="sum"), torch.tensor(labels.numel())

    def unreduced_call(
        self, inputs: Tensor, labels: Tensor, exp_info: ExpInfo
    ) -> Tensor:
        return l1_loss(inputs, labels, reduction="none")


class CrossEntropy(LossLikeMetric):

    def smaller_is_better(self) -> bool:
        return True

    def is_scalar(self) -> bool:
        return True

    def __call__(
        self, inputs: Tensor, labels: Tensor, exp_info: ExpInfo
    ) -> Tuple[Tensor, Tensor]:
        return cross_entropy(inputs, labels, reduction="sum"), torch.tensor(
            labels.numel()
        )

    def unreduced_call(
        self, inputs: Tensor, labels: Tensor, exp_info: ExpInfo
    ) -> Tensor:
        return cross_entropy(inputs, labels, reduction="none")


class Accuracy(LossLikeMetric):

    def smaller_is_better(self) -> bool:
        return False

    def is_scalar(self) -> bool:
        return True

    def unreduced_call(
        self, inputs: Tensor, labels: Tensor, exp_info: ExpInfo
    ) -> Tensor:
        return torch.argmax(inputs, dim=1) == labels

    def __call__(
        self, inputs: Tensor, labels: Tensor, exp_info: ExpInfo
    ) -> Tuple[Tensor, Tensor]:
        acc = self.unreduced_call(inputs, labels, exp_info)
        return torch.sum(acc.float()), torch.tensor(labels.numel())


def _groupby_sum(inputs: Tensor, classes, num_classes) -> Tuple[Tensor, Tensor]:
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
    label_counts = label_counts.scatter_add_(
        0, classes, torch.ones_like(inputs, dtype=label_counts.dtype)
    )

    sum_by_class = torch.zeros(num_classes, dtype=inputs.dtype, device=classes.device)
    sum_by_class = sum_by_class.scatter_add_(dim=0, index=classes, src=inputs)

    return sum_by_class, label_counts


def _split_frequencies_by_groups(sorted_labels, freq_sorted, n_splits):
    cum_freq_sorted = freq_sorted.cumsum(0)
    freq_breakpoints = torch.linspace(0, 1, n_splits + 1, device=freq_sorted.device)[1:-1]
    indices = torch.searchsorted(cum_freq_sorted, freq_breakpoints, side="left")

    split_sizes = []
    previous_idx = 0
    for idx in indices:
        split_sizes.append((1 + idx - previous_idx).item())
        previous_idx = idx
    split_sizes.append(len(sorted_labels) - sum(split_sizes))

    splits = torch.split(sorted_labels, split_size_or_sections=split_sizes, dim=0)
    return splits


@frozen
class PerClass(LossLikeMetric):
    metric: LossLikeMetric
    groups: int = 10

    def smaller_is_better(self) -> bool:
        return self.metric.smaller_is_better()

    def is_scalar(self) -> bool:
        return False

    def unreduced_call(
        self, inputs: Tensor, labels: Tensor, exp_info: ExpInfo
    ) -> Tensor:
        return self.metric.unreduced_call(inputs, labels, exp_info)


    def _group_unreduced_call(self, values, labels, class_counts):
        num_classes = len(class_counts)
        sum_by_class, counts = _groupby_sum(values, labels, num_classes)

        sort_idx = torch.flip(class_counts.argsort(), dims=[0])
        all_labels = torch.arange(0, num_classes, device=class_counts.device)
        sorted_labels = all_labels[sort_idx]
        freq_sorted = class_counts[sort_idx] / class_counts.sum()
        groups = _split_frequencies_by_groups(sorted_labels, freq_sorted, self.groups)

        losses_per_group = torch.stack([torch.sum(sum_by_class[g]) for g in groups])
        counts_per_group = torch.stack([torch.sum(counts[g]) for g in groups])

        return losses_per_group, counts_per_group


    def __call__(
        self, inputs: Tensor, labels: Tensor, exp_info: ExpInfo
    ) -> Tuple[Tensor, Tensor]:
        dataset = exp_info.exp.problem.dataset
        if not isinstance(dataset, HasClassCounts):
            raise ValueError(
                f"Asked to compute PerClassMetric {self} on dataset {dataset}. "
                "But dataset does not have class counts"
            )

        class_counts = dataset.class_counts("tr")
        out_shape = exp_info.exp.problem.dataset.model_output_shape(inputs.shape[0])
        num_classes = out_shape[-1]

        assert class_counts.numel() == num_classes
        assert len(class_counts.shape) == 1

        values = self.metric.unreduced_call(inputs, labels, exp_info)
        values = values.to(torch.float)
        return self._group_unreduced_call(values, labels, class_counts)


    def plot_label(self) -> str:
        return self.metric.plot_label() + " Per Class"


class CrossEntropyPerClass(LossLikeMetric):
    """Cross entropy loss per class.

    Can result in large logs on problems with many classes.
    """

    def __call__(self, inputs, labels, exp_info: ExpInfo):
        warnings.warn(
            "CrossEntropyPerClass is deprecated. "
            "Use PerClass(CrossEntropy()) instead for new experiments."
        )
        num_classes = inputs.shape[1]
        losses = cross_entropy(inputs, labels, reduction="none")
        return _groupby_sum(losses, labels, num_classes)

    def smaller_is_better(self) -> bool:
        return True

    def is_scalar(self):
        return False

    def unreduced_call(
        self, inputs: torch.Tensor, labels: torch.Tensor, exp_info: ExpInfo
    ) -> torch.Tensor:
        return CrossEntropy().unreduced_call(inputs, labels, exp_info)


class AccuracyPerClass(LossLikeMetric):
    """Accuracy per class.

    Can result in large logs on problems with many classes.
    """

    def __call__(self, inputs, labels, exp_info: ExpInfo):
        warnings.warn(
            "AccuracyPerClass is deprecated. "
            "Use PerClass(Accuracy()) instead for new experiments."
        )
        num_classes = inputs.shape[1]
        classes = torch.argmax(inputs, dim=1)
        accuracy_per_sample = (classes == labels).float()
        return _groupby_sum(accuracy_per_sample, labels, num_classes)

    def smaller_is_better(self) -> bool:
        return False

    def is_scalar(self):
        return False

    def unreduced_call(
        self, inputs: torch.Tensor, labels: torch.Tensor, exp_info: ExpInfo
    ) -> torch.Tensor:
        return Accuracy().unreduced_call(inputs, labels, exp_info)
