import torch

from optexp import Experiment, Problem
from optexp.datasets import DummyRegression
from optexp.hardwareconfig import StrictManualConfig
from optexp.metrics import Accuracy, CrossEntropy
from optexp.models.linear import Linear
from optexp.optim import SGD
from optexp.runner.runner import run_experiment


def make_toy_experiment(lr, batch_size, device, seed):
    return Experiment(
        optim=SGD(lr=lr),
        problem=Problem(
            dataset=DummyRegression(),
            model=Linear(),
            lossfunc=CrossEntropy(),
            metrics=[CrossEntropy(), Accuracy()],
            batch_size=batch_size,
        ),
        group="testing",
        eval_every=10,
        seed=seed,
        steps=50,
        hardware_config=StrictManualConfig(
            device=device,
        ),
    )


def test_run_without_crash():
    run_experiment(make_toy_experiment(lr=0.1, batch_size=5, device="cuda", seed=0))
