from typing import List

from optexp.cli import ExpGroup, cli
from optexp.datasets import DummyRegression
from optexp.experiment import Experiment
from optexp.hardwareconfig.strict_manual import StrictManualConfig
from optexp.metrics.metrics import MSE
from optexp.models import Linear
from optexp.optim.sgd import SGD
from optexp.problem import Problem
from optexp.runner.slurm.slurm_config import SlurmConfig


def make_experiments(momentum, group_name) -> List[Experiment]:
    return [
        Experiment(
            optim=SGD(lr=lr, momentum=momentum),
            problem=Problem(
                dataset=DummyRegression(),
                model=Linear(),
                lossfunc=MSE(),
                metrics=[MSE()],
                batch_size=100,
            ),
            group=group_name,
            eval_every=1,
            seed=0,
            steps=100,
            hardware_config=StrictManualConfig(
                num_devices=1,
                micro_batch_size=10,
                eval_micro_batch_size=10,
                device="cpu",
            ),
        )
        for lr in [10**i for i in [-10]]
    ]


SLURM_CONFIG = SlurmConfig(hours=1, gb_ram=8, n_cpus=1, n_gpus=1, gpu=True)

groups_and_params = {
    "momentum_0": 0,
    "momentum_0.9": 0.9,
}

exp_groups = {
    group_name: ExpGroup(
        exps=make_experiments(momentum, group_name),
        slurm_config=SLURM_CONFIG,
    )
    for group_name, momentum in groups_and_params.items()
}

if __name__ == "__main__":
    cli(exp_groups)
