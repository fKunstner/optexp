import numpy as np
import torch

from optexp.optim import SGD


def test_numpy_and_pytorch_agree_on_representation():

    lrs_numpy = np.logspace(-15, 15, base=10, num=150)
    lrs_pytorch = torch.logspace(-15, 15, base=10, steps=150)

    for lr_np, lr_to in zip(lrs_numpy, lrs_pytorch):
        opt1 = SGD(lr=lr_np)
        opt2 = SGD(lr=lr_to.item())
        assert opt1.equivalent_definition() == opt2.equivalent_definition()
