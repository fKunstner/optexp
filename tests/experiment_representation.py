import dataclasses

import pytest

from optexp import Experiment, Problem
from optexp.datasets import DummyRegression
from optexp.hardwareconfig import StrictManualConfig
from optexp.metrics import Accuracy, CrossEntropy
from optexp.models.linear import Linear
from optexp.optim import SGD

BASE_EXPERIMENT = Experiment(
    optim=SGD(lr=1.0),
    problem=Problem(
        dataset=DummyRegression(),
        model=Linear(),
        lossfunc=CrossEntropy(),
        metrics={Accuracy(), CrossEntropy()},
        batch_size=100,
    ),
    steps=0,
    eval_every=10,
)

experiments = [
    BASE_EXPERIMENT,
    dataclasses.replace(
        BASE_EXPERIMENT, hardware_config=StrictManualConfig(device="cpu")
    ),
    dataclasses.replace(
        BASE_EXPERIMENT, hardware_config=StrictManualConfig(device="cuda")
    ),
    dataclasses.replace(
        BASE_EXPERIMENT,
        hardware_config=StrictManualConfig(
            num_devices=2, micro_batch_size=10, eval_micro_batch_size=10, num_workers=4
        ),
    ),
    dataclasses.replace(
        BASE_EXPERIMENT,
        problem=dataclasses.replace(
            BASE_EXPERIMENT.problem, metrics={CrossEntropy(), Accuracy()}
        ),
    ),
]


def test_repr():
    exp1 = BASE_EXPERIMENT
    exp1_repr = eval(repr(exp1))
    assert exp1 == exp1_repr
    assert exp1.equivalent_hash() == exp1_repr.equivalent_hash()


def test_equivalent_form_repr():
    exp1 = BASE_EXPERIMENT
    exp1_repr = eval(repr(exp1))
    assert exp1.equivalent_definition() == exp1_repr.equivalent_definition()
    assert exp1.equivalent_hash() == exp1_repr.equivalent_hash()


@pytest.mark.parametrize("exp", experiments)
def test_equivalent_form(exp):
    assert exp.equivalent_definition() == BASE_EXPERIMENT.equivalent_definition()
    assert exp.equivalent_hash() == BASE_EXPERIMENT.equivalent_hash()
