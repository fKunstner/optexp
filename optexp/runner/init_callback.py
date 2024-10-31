from abc import ABC
from typing import TYPE_CHECKING, Callable, Iterable

from attr import field, frozen

from optexp.component import Component

if TYPE_CHECKING:
    from optexp.experiment import Experiment
    from optexp.runner.exp_state import ExperimentState


@frozen
class InitCallback(Component, ABC):
    def __call__(
        self,
        exp: "Experiment",
        exp_state: "ExperimentState",
        log: Callable[[str], None],
    ) -> "ExperimentState":
        raise NotImplementedError


@frozen
class InitCallbackSequence(InitCallback):
    callbacks: Iterable[InitCallback] = field(converter=tuple)

    def __call__(
        self,
        exp: "Experiment",
        exp_state: "ExperimentState",
        log: Callable[[str], None],
    ) -> "ExperimentState":
        for callback in self.callbacks:
            exp_state = callback(exp, exp_state, log)
        return exp_state
