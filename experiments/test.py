from fractions import Fraction

import torch

from optexp.cli import exp_runner_cli
from optexp.datasets.dataset import epoch_to_steps
from optexp.datasets.image import MNIST
from optexp.experiments.experiment import Experiment
from optexp.experiments.hardwareconfig import RawHardwareConfig
from optexp.models.vision import LeNet5
from optexp.optimizers import SGD, LearningRate
from optexp.problems import Problem
from optexp.problems.metrics import Accuracy, CrossEntropyLoss
from optexp.slurm.slurm_config import SlurmConfig

problem = EPOCHS = 1
group = "testing"

dataset = MNIST()
batch_size = 1000
steps = 1  # epoch_to_steps(epochs=100, dataset=dataset, batch_size=100)

experiments = [
    Experiment(
        optim=SGD(lr=LearningRate(Fraction(1, 2))),
        problem=Problem(
            dataset=dataset,
            model=LeNet5(),
            lossfunc=torch.nn.CrossEntropyLoss,
            metrics=[CrossEntropyLoss, Accuracy],
            batch_size=batch_size,
        ),
        group=group,
        eval_every=1,
        batch_size=batch_size,
        seed=0,
        steps=steps,
        hw_config=RawHardwareConfig(
            num_workers=1,
            micro_batch_size=batch_size,
            eval_micro_batch_size=batch_size,
            device="cpu",
            wandb_autosync=False,
        ),
    )
]

SLURM_CONFIG = SlurmConfig(hours=10, gb_ram=8, n_cpus=1, n_gpus=1, gpu=True)
if __name__ == "__main__":
    exp_runner_cli(experiments, slurm_config=SLURM_CONFIG)
