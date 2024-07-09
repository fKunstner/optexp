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
        steps=0,
        hardware_config=StrictManualConfig(
            device=device,
        ),
    )


def test_seed_initialization():
    exp1 = make_toy_experiment(lr=0.1, batch_size=10, device="cpu", seed=0)
    exp2 = make_toy_experiment(lr=0.1, batch_size=5, device="cuda", seed=0)

    state1 = run_experiment(exp1)
    state2 = run_experiment(exp2)

    for p1, p2 in zip(state1.model.parameters(), state2.model.parameters()):
        assert torch.allclose(p1, p2.to(p1.device))
