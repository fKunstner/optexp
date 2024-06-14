import math
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Tuple

import torch

from optexp import config
from optexp.config import get_device, get_logger
from optexp.datasets import Dataset
from optexp.models import Model
from optexp.problems.exceptions import DivergingException


@dataclass
class Problem:
    """Wrapper for a model and dataset defining a problem to optimize.

    Attributes:
        model: The model that will be optimized.
        dataset: The dataset to use.
    """

    model: Model
    dataset: Dataset
    batch_size: int
    lossfunc: torch.nn.Module
    metrics: List[torch.nn.Module]
    micro_batch_size: Literal["auto"] | int = "auto"

    def init_problem(self) -> None:
        """Loads the dataset and the PyTorch model onto device."""
        get_logger().info("Loading problem: " + self.__class__.__name__)

        self.train_loader = self.dataset.load(b=self.batch_size, tr_va="tr")
        self.val_loader = self.dataset.load(b=self.batch_size, tr_va="va")
        self.input_shape = self.dataset.input_shape(self.batch_size)
        self.output_shape = self.dataset.output_shape(self.batch_size)

        self.torch_model = self.model.load_model(self.input_shape, self.output_shape)
        self.torch_model = self.torch_model.to(get_device())

        self.criterion = self.init_loss()

    def eval_raw(
        self, val: bool = True
    ) -> Tuple[
        int, Dict[torch.nn.Module, torch.Tensor], Dict[torch.nn.Module, torch.Tensor]
    ]:
        num_samples = 0
        running_metrics: Dict[torch.nn.Module, torch.Tensor] = {}
        running_n_samples: Dict[torch.nn.Module, torch.Tensor] = {}

        def add_(d, k, v):
            if k in d:
                d[k] += v
            else:
                d[k] = v
            return d

        if val:
            loader = self.val_loader
        else:
            loader = self.train_loader

        with torch.no_grad():
            for _, (features, labels) in enumerate(loader):
                features = features.to(config.get_device())
                labels = labels.to(config.get_device())

                y_pred = self.torch_model(features)
                for module in self.get_criterions():
                    outputs = module(y_pred, labels)
                    if type(outputs) is not tuple:
                        value = outputs.detach()

                        if math.isnan(value) or math.isinf(value):
                            raise DivergingException(
                                f"{str(module)[:-2]} ({'va' if val else 'tr'})is NAN or INF."
                            )
                        add_(running_metrics, module, value * len(features))
                    else:
                        value, weight = outputs[0].detach(), outputs[1].detach()
                        add_(running_metrics, module, value)
                        add_(running_n_samples, module, weight)

                num_samples += len(features)

                del y_pred
                del features
                del labels
        return num_samples, running_metrics, running_n_samples

    def eval(self, val: bool = True) -> dict:
        """Wrapper to evaluate model. Provides the list of criterions to use.

        Args:
            val (bool, optional): When True model is evaluated on validation dataset,
                otherwise training dataset. Defaults to True.

        Returns:
            A dictionary where keys are strings representing the name of the criterions and values
            are the accumulated values of the criterions on the entire dataset.
        """
        criterions = self.get_criterions()
        num_samples, running_metrics, running_n_samples = self.eval_raw(val=val)
        key_prefix = "va" if val else "tr"

        for module in criterions:
            if torch.numel(running_metrics[module]) > 1:
                running_metrics[module] = (
                    running_metrics[module] / running_n_samples[module]
                )
            else:
                running_metrics[module] /= num_samples

        metrics: Dict[str, Any] = {
            f"{key_prefix}_{str(module)[:-2]}": (
                value.tolist() if torch.numel(value) > 1 else value.item()
            )
            for module, value in running_metrics.items()
        }

        return metrics

    def one_epoch(self, optimizer: torch.optim.Optimizer) -> dict:
        """Optimizes the model on a specific loss function defined for this
        problem for one epoch on the training set.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer to use for updating model parameters

        Raises:
            DivergingException: Raised when the value for any of the metrics is NAN or INF.

        Returns:
            Keys are strings representing the name of the criterions and values
            are the accumulated values of the criterion on the entire training dataset.
        """
        needs_closure = "lr" not in optimizer.defaults
        train_loss = 0.0
        num_samples = 0
        mini_batch_losses = []
        for _, (features, labels) in enumerate(self.train_loader):
            features = features.to(config.get_device())
            labels = labels.to(config.get_device())

            def closure():
                y_pred = self.torch_model(features)
                out = self.criterion(y_pred, labels)

                if type(out) is tuple:
                    loss, weight = out
                    loss = loss / weight
                else:
                    loss = out

                if math.isnan(loss) or math.isinf(loss):
                    raise DivergingException("Live training loss is NAN or INF.")

                return loss

            optimizer.zero_grad()

            loss = closure()
            loss.backward()
            optimizer.step(closure=closure if needs_closure else None)

            mini_batch_losses.append(loss.item())
            train_loss += loss.item() * len(features)
            num_samples += len(features)

        metrics = {
            "live_train_loss": train_loss / num_samples,
            "mini_batch_losses": mini_batch_losses,
        }
        return metrics

    def init_loss(self) -> torch.nn.Module:
        return self.lossfunc

    def get_criterions(self) -> List[torch.nn.Module]:
        return self.metrics
