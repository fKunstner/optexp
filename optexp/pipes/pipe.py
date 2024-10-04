from abc import ABC, abstractmethod
from typing import Tuple

from torch import Tensor

from optexp.component import Component
from optexp.datasets.dataset import Split
from optexp.metrics import LossLikeMetric, Metric
from optexp.metrics.metric import GraphMetric


class DataPipe(Component, ABC):

    @abstractmethod
    def forward(self, data, model, split: Split):
        raise NotImplementedError()

    def forward_or_cache(self, data, model, split: Split, cached_forward=None):
        if cached_forward is not None:
            return cached_forward
        return self.forward(data, model, split)

    @abstractmethod
    def compute_loss(  # pylint: disable=too-many-arguments
        self, data, model, lossfunc, split: Split, cached_forward=None
    ) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError()

    @abstractmethod
    def compute_metric(  # pylint: disable=too-many-arguments
        self, data, model, metric: Metric, split: Split, cached_forward=None
    ) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError()


class TensorDataPipe(DataPipe):
    @staticmethod
    def _check_data(data):
        is_tensor_data = (
            isinstance(data, (tuple, list))
            and len(data) == 2
            and all(isinstance(d, Tensor) for d in data)
        )
        if not is_tensor_data:
            raise ValueError(
                "Unknown data type. "
                "Expected tuple[Tensor, Tensor] or list[Tensor] for tensor data "
                f"but tot {type(data)}."
                "Did you select the correct DataPipe?"
            )

    def forward(self, data, model, split: Split):
        self._check_data(data)
        return model(data[0])

    def compute_metric(  # pylint: disable=too-many-arguments
        self, data, model, metric: Metric, split: Split, cached_forward=None
    ) -> Tuple[Tensor, Tensor]:
        forward = self.forward_or_cache(data, model, split, cached_forward)
        if isinstance(metric, LossLikeMetric):
            return metric(forward, data[1])
        raise ValueError(
            f"Unknown metric type: {type(metric)}. " f"Expected LossLikeMetric."
        )

    def compute_loss(  # pylint: disable=too-many-arguments
        self, data, model, lossfunc, split: Split, cached_forward=None
    ) -> Tuple[Tensor, Tensor]:
        forward = self.forward_or_cache(data, model, split, cached_forward)
        return lossfunc(forward, data[1])


class GraphDataPipe(DataPipe):
    @staticmethod
    def _check_data(data):
        is_graph_data = all(
            hasattr(data, name)
            for name in ["x", "y", "edge_index", "train_mask", "val_mask"]
        )
        if not is_graph_data:
            raise ValueError(
                "Unknown data type. "
                "Expected torch_geometric.data.Data for graph data "
                f"but got {type(data)}."
                "Did you select the correct DataPipe?"
            )

    def forward(self, data, model, split: Split):
        self._check_data(data)
        return model(data.x, data.edge_index)

    def compute_metric(  # pylint: disable=too-many-arguments
        self, data, model, metric: Metric, split: Split, cached_forward=None
    ) -> Tuple[Tensor, Tensor]:
        model_out = self.forward_or_cache(data, model, split, cached_forward)
        if split == "tr":
            mask = data.train_mask
        elif split == "va":
            mask = data.val_mask
        else:
            mask = data.test_mask
        if isinstance(metric, LossLikeMetric):
            return metric(model_out[mask], data.y[mask])
        if isinstance(metric, GraphMetric):
            return metric(data, mask, model_out[mask], data.y[mask])
        raise ValueError(
            f"Unknown metric type: {type(metric)}. "
            f"Expected LossLikeMetric or InputOutputLabelMetric."
        )

    def compute_loss(  # pylint: disable=too-many-arguments
        self, data, model, lossfunc, split: Split, cached_forward=None
    ) -> Tuple[Tensor, Tensor]:
        model_out = self.forward_or_cache(data, model, split, cached_forward)
        if split == "tr":
            mask = data.train_mask
        elif split == "va":
            mask = data.val_mask
        else:
            mask = data.test_mask
        return lossfunc(model_out[mask], data.y[mask])
