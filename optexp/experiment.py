from attrs import field, frozen

from optexp.component import Component
from optexp.hardwareconfig import StrictManualConfig
from optexp.hardwareconfig.hardwareconfig import HardwareConfig
from optexp.optim.optimizer import Optimizer
from optexp.problem import Problem


@frozen
class Experiment(Component):
    """Specify an experiment.

    Args:
         optim (Optimizer): optimizer to use.
         problem (Problem): problem to solve.
         eval_every (int): often to evaluate the metrics.
         steps (int): total number of steps.
            To convert from epochs, use :func:`optexp.utils.epochs_to_steps`.
         seed (int, optional): seed for the random number generator.
            Defaults to 0.
         hardware_config (HardwareConfig, optional): implementation details.
            Defaults to :class:`~optexp.hardwareconfig.StrictManualConfig()`.
         group (str, optional): name for logging. Defaults to ``"default"``.
    """

    optim: Optimizer
    problem: Problem
    eval_every: int
    steps: int
    seed: int = 0
    hardware_config: HardwareConfig = field(
        default=StrictManualConfig(), repr=False, hash=False
    )
    group: str = field(default="default", repr=False, hash=False)
