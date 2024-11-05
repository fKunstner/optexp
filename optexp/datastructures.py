from typing import TYPE_CHECKING, Optional

from attr import frozen

if TYPE_CHECKING:
    from optexp import Experiment
    from optexp.datasets.dataset import Split
    from optexp.runner.exp_state import ExperimentState


@frozen
class AdditionalInfo:
    """Additional information potentially required by metrics."""

    split: "Split"
    exp: "Experiment"
    exp_state: "ExperimentState"
    cached_forward: Optional[object] = None


@frozen
class ExpInfo:
    exp: "Experiment"
    exp_state: "ExperimentState"
