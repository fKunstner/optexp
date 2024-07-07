import torch

from optexp.cli import cli
from optexp.datasets.mnist import MNIST
from optexp.experiment import Experiment
from optexp.hardwareconfig.strict_manual import StrictManualConfig
from optexp.metrics.metrics import Accuracy, CrossEntropyLoss
from optexp.models.vision import LeNet5
from optexp.optim.sgd import SGD
from optexp.problem import Problem
from optexp.runner.slurm.slurm_config import SlurmConfig

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
        hardware_config=StrictManualConfig(
            num_devices=1,
            micro_batch_size=1000,
            eval_micro_batch_size=1000,
            device="cpu",
        ),
    )
]

SLURM_CONFIG = SlurmConfig(hours=10, gb_ram=8, n_cpus=1, n_gpus=1, gpu=True)
if __name__ == "__main__":
    cli(experiments, slurm_config=SLURM_CONFIG)
