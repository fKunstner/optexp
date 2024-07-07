import operator
from functools import reduce

import torch

from optexp.models import Model
from optexp.models.model import assert_batch_sizes_match


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
