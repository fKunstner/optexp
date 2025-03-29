import operator
from functools import reduce

import torch
from attr import frozen

from optexp.models.model import Model, assert_batch_sizes_match


@frozen
class Linear(Model):
    """A linear model for regression or classification.

    Can take inputs of any shape, and will flatten them first.

    Args:
        bias (bool, optional): whether to include a bias term. Defaults to True.
    """

    bias: bool = True

    def load_model(self, input_shape, output_shape):
        b1, dim_inp = input_shape[0], reduce(operator.mul, input_shape[1:], 1)
        b2, dim_out = output_shape
        assert_batch_sizes_match(b1, b2)

        return torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(dim_inp, dim_out, bias=self.bias),
        )


@frozen
class ReLUMLP(Model):
    """A linear model for regression or classification.

    Can take inputs of any shape, and will flatten them first.

    Args:
        bias (bool, optional): whether to include a bias term. Defaults to True.
    """

    bias: bool = True
    hidden_dimensions: list = [128]

    def load_model(self, input_shape, output_shape):
        b1, dim_inp = input_shape[0], reduce(operator.mul, input_shape[1:], 1)
        b2, dim_out = output_shape
        assert_batch_sizes_match(b1, b2)
        dimensions = [dim_inp] + self.hidden_dimensions + [dim_out]
        modules = [torch.nn.Flatten()]
        for i in range(len(dimensions) - 1):
            modules.append(
                torch.nn.Linear(dimensions[i], dimensions[i + 1], bias=self.bias)
            )
            if i < len(dimensions) - 2:
                modules.append(torch.nn.ReLU())

        return torch.nn.Sequential(*modules)
