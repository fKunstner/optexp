from optexp.cli import cli
from optexp.datasets import DummyRegression
from optexp.experiment import Experiment
from optexp.hardwareconfig.strict_manual import StrictManualConfig
from optexp.metrics.metrics import MSE, Accuracy, CrossEntropy
from optexp.models import Linear
from optexp.optim.sgd import SGD
from optexp.problem import Problem
from optexp.runner.slurm.slurm_config import SlurmConfig

experiments = [
    Experiment(
        optim=SGD(lr=lr, momentum=momentum),
        problem=Problem(
            dataset=DummyRegression(),
            model=Linear(),
            lossfunc=MSE(),
            metrics=[MSE()],
            batch_size=100,
        ),
        group="testing",
        eval_every=1,
        seed=seed,
        steps=100,
        hardware_config=StrictManualConfig(
            num_devices=1,
            micro_batch_size=10,
            eval_micro_batch_size=10,
            device="cpu",
        ),
    )
    for lr in [10**i for i in [-6, -5, -4, -3, -2, -1, 0, 1, 2]]
    for momentum in [0, 0.9]
    for seed in [0, 1, 2]
]

SLURM_CONFIG = SlurmConfig(hours=1, gb_ram=8, n_cpus=1, n_gpus=1, gpu=True)
if __name__ == "__main__":
    cli(experiments, slurm_config=SLURM_CONFIG)
