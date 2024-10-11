from abc import ABC
from typing import TYPE_CHECKING, Iterable

from attr import field, frozen

from optexp.component import Component

if TYPE_CHECKING:
    from optexp.experiment import Experiment
    from optexp.runner.exp_state import ExperimentState


@frozen
class InitCallback(Component, ABC):
    def __call__(
        self, exp: "Experiment", exp_state: "ExperimentState"
    ) -> "ExperimentState":
        raise NotImplementedError


@frozen
class InitCallbackSequence(InitCallback):
    callbacks: Iterable[InitCallback] = field(converter=tuple)

    def __call__(
        self, exp: "Experiment", exp_state: "ExperimentState"
    ) -> "ExperimentState":
        for callback in self.callbacks:
            exp_state = callback(exp, exp_state)
        return exp_state
