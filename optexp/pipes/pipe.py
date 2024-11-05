from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import Tensor

from optexp.component import Component
from optexp.datastructures import AdditionalInfo, ExpInfo
from optexp.metrics import LossLikeMetric, Metric
from optexp.metrics.metric import GraphLossLikeMetric


def exp_info(additional_info: AdditionalInfo) -> ExpInfo:
    return ExpInfo(additional_info.exp, additional_info.exp_state)


class DataPipe(Component, ABC):

    @abstractmethod
    def forward(self, data, model):
        raise NotImplementedError()

    def forward_or_cache(self, data, model, cached_forward):
        if cached_forward is not None:
            return cached_forward
        return self.forward(data, model)

    @abstractmethod
    def compute_loss(
        self, data, model, lossfunc, additional_info: AdditionalInfo
    ) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError()

    @abstractmethod
    def compute_metric(
        self, data, model, metric: LossLikeMetric, additional_info: AdditionalInfo
    ) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError()


class TensorDataPipe(DataPipe):
    """DataPipe for typical tensor data, where the inputs (outputs) are of shape [n, d] ([n])."""

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

    def forward(self, data, model):
        self._check_data(data)
        return model(data[0])

    def compute_metric(
        self, data, model, metric: LossLikeMetric, additional_info: AdditionalInfo
    ) -> Tuple[Tensor, Tensor]:
        forward = self.forward_or_cache(data, model, additional_info.cached_forward)
        return metric(forward, data[1], exp_info(additional_info))

    def compute_loss(
        self, data, model, lossfunc, additional_info: AdditionalInfo
    ) -> Tuple[Tensor, Tensor]:
        forward = self.forward_or_cache(data, model, additional_info.cached_forward)
        return lossfunc(forward, data[1])


class SequenceDataPipe(DataPipe):
    """DataPipe for sequence data, used in language models.

    Inputs and outputs are both Long tensors of shape [n_sequences, sequence_length]
    but the output of the model is of shape [n_sequences, sequence_length, vocab_size]
    and the first two dimensions need to be flattened before computing the loss.
    """

    @staticmethod
    def _check_data(data):
        is_tensor_data = (
            isinstance(data, (tuple, list))
            and len(data) == 2
            and all(isinstance(d, Tensor) for d in data)
            and all(d.dtype == torch.long for d in data)
        )
        if not is_tensor_data:
            raise ValueError(
                "Unknown data type. "
                "Expected tuple[Tensor, Tensor] or list[Tensor] containing long for sequence data "
                f"but tot {type(data)}. "
                "Did you select the correct DataPipe?"
            )

    def forward(self, data, model):
        self._check_data(data)
        return model(data[0])

    def compute_metric(
        self, data, model, metric: LossLikeMetric, additional_info: AdditionalInfo
    ) -> Tuple[Tensor, Tensor]:
        forward = self.forward_or_cache(data, model, additional_info.cached_forward)
        return metric(
            forward.reshape(-1, forward.shape[2]),
            data[1].reshape(-1),
            exp_info(additional_info),
        )

    def compute_loss(
        self, data, model, lossfunc, additional_info: AdditionalInfo
    ) -> Tuple[Tensor, Tensor]:
        forward = self.forward_or_cache(data, model, additional_info.cached_forward)
        return lossfunc(forward.reshape(-1, forward.shape[2]), data[1].reshape(-1))


def get_mask(data, additional_info):
    if additional_info.split == "tr":
        return data.train_mask
    if additional_info.split == "va":
        return data.val_mask
    return data.test_mask


class GraphDataPipe(DataPipe):
    """DataPipe for graph data, where the data is a torch_geometric.data.Data object."""

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

    def forward(self, data, model):
        self._check_data(data)
        return model(data.x, data.edge_index)

    def compute_metric(
        self, data, model, metric: Metric, additional_info: AdditionalInfo
    ) -> Tuple[Tensor, Tensor]:
        model_out = self.forward_or_cache(data, model, additional_info.cached_forward)
        mask = get_mask(data, additional_info)
        if isinstance(metric, GraphLossLikeMetric):
            return metric(
                data, mask, model_out[mask], data.y[mask], exp_info(additional_info)
            )
        return metric(model_out[mask], data.y[mask], exp_info(additional_info))

    def compute_loss(
        self, data, model, lossfunc, additional_info: AdditionalInfo
    ) -> Tuple[Tensor, Tensor]:
        model_out = self.forward_or_cache(data, model, additional_info.cached_forward)
        mask = get_mask(data, additional_info)
        return lossfunc(model_out[mask], data.y[mask])
