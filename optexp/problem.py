from typing import Iterable

from attrs import field, frozen

from optexp.component import Component
from optexp.datasets.dataset import Dataset
from optexp.metrics.metric import Metric
from optexp.models.model import Model


@frozen
class Problem(Component):
    """Specify a problem.

    Args:
        model (Model): model to optimize.
        dataset (Dataset): dataset to fit the model to.
        batch_size (int): effective batch size.
           To use gradient accumulation, set the ``micro_batch_size``
           in :class:`optexp.hardwareconfig.HardwareConfig`.
        lossfunc (Metric): loss function to use for optimization.
        metrics (Iterable[Metric]): metrics to evaluate.
    """

    model: Model
    dataset: Dataset
    batch_size: int
    lossfunc: Metric
    metrics: Iterable[Metric] = field(converter=frozenset)
