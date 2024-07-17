from abc import ABC, abstractmethod
from typing import Tuple

from torch import Tensor

from optexp.component import Component
from optexp.datasets.dataset import TrVa
from optexp.metrics import Metric


class DataPipe(Component, ABC):

    @abstractmethod
    def forward(self, data, model, trva: TrVa):
        raise NotImplementedError()

    @abstractmethod
    def compute_loss(
        self, data, model, lossfunc, trva: TrVa, cached_forward=None
    ) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError()

    @abstractmethod
    def compute_metric(
        self, data, model, metric: Metric, trva: TrVa, cached_forward=None
    ) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError()


class TensorDataPipe(DataPipe):
    def _check_data(self, data):
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

    def forward(self, data, model, trva: TrVa):
        self._check_data(data)
        return model(data[0])

    def compute_metric(  # pylint: disable=too-many-arguments
        self,
        data,
        model,
        metric: Metric,
        trva: TrVa,
        cached_forward=None,
    ) -> Tuple[Tensor, Tensor]:
        forward = (
            self.forward(data, model, trva)
            if cached_forward is None
            else cached_forward
        )
        return metric(forward, data[1])

    def compute_loss(
        self, data, model, lossfunc, trva: TrVa, cached_forward=None
    ) -> Tuple[Tensor, Tensor]:
        forward = (
            self.forward(data, model, trva)
            if cached_forward is None
            else cached_forward
        )
        return lossfunc(forward, data[1])


class GraphDataPipe(DataPipe):
    def _check_data(self, data):
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

    def forward(self, data, model, trva: TrVa):
        self._check_data(data)
        return model(data.x, data.edge_index)

    def compute_metric(
        self, data, model, metric: Metric, trva: TrVa, cached_forward=None
    ) -> Tuple[Tensor, Tensor]:
        model_out = (
            self.forward(data, model, trva)
            if cached_forward is None
            else cached_forward
        )
        mask = data.train_mask if trva == "tr" else data.val_mask
        return metric(model_out[mask], data.y[mask])

    def compute_loss(
        self, data, model, lossfunc, trva: TrVa, cached_forward=None
    ) -> Tuple[Tensor, Tensor]:
        model_out = (
            self.forward(data, model, trva)
            if cached_forward is None
            else cached_forward
        )
        mask = data.train_mask if trva == "tr" else data.val_mask
        return lossfunc(model_out[mask], data.y[mask])
