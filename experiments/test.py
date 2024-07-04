import torch

from optexp.cli import exp_runner_cli
from optexp.components.datasets import MNIST
from optexp.components.experiment import Experiment
from optexp.components.hardwareconfigs.strict_manual import StrictManualDetails
from optexp.components.metrics.metrics import Accuracy, CrossEntropyLoss
from optexp.components.models import LeNet5
from optexp.components.optimizers.sgd import SGD
from optexp.components.problem import Problem
from optexp.slurm.slurm_config import SlurmConfig

experiments = [
    Experiment(
        optim=SGD(lr=10**0.5),
        problem=Problem(
            dataset=MNIST(),
            model=LeNet5(),
            lossfunc=torch.nn.CrossEntropyLoss,
            metrics=[CrossEntropyLoss, Accuracy],
            batch_size=1000,
        ),
        group="testing",
        eval_every=1,
        seed=0,
        steps=1,
        implementation=StrictManualDetails(
            num_workers=1,
            micro_batch_size=1000,
            eval_micro_batch_size=1000,
            device="cpu",
        ),
    )
]

SLURM_CONFIG = SlurmConfig(hours=10, gb_ram=8, n_cpus=1, n_gpus=1, gpu=True)
if __name__ == "__main__":
    exp_runner_cli(experiments, slurm_config=SLURM_CONFIG)
