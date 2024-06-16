from fractions import Fraction

import torch

from optexp.cli import exp_runner_cli
from optexp.datasets.image import MNIST
from optexp.experiments.experiment import Experiment
from optexp.models.vision import LeNet5
from optexp.optimizers import SGD, LearningRate
from optexp.problems import Problem
from optexp.problems.metrics import Accuracy, CrossEntropyLoss
from optexp.slurm.slurm_config import SlurmConfig

dataset = MNIST()
model = LeNet5()
problem = Problem(
    model,
    dataset,
    batch_size=20000,
    lossfunc=torch.nn.CrossEntropyLoss,
    metrics=[CrossEntropyLoss, Accuracy],
)
EPOCHS = 1
group = "testing"
experiments = [
    Experiment(
        optim=SGD(lr=LearningRate(Fraction(1, 2))),
        problem=problem,
        epochs=0,
        steps=1,
        group=group,
        seed=0,
    )
]

SLURM_CONFIG = SlurmConfig(hours=10, gb_ram=8, n_cpus=1, n_gpus=1, gpu=True)
if __name__ == "__main__":
    exp_runner_cli(experiments, slurm_config=SLURM_CONFIG)
