from torch.nn import CrossEntropyLoss

from optexp import Experiment, Problem, cli, metrics
from optexp.datasets import MNIST
from optexp.hardwareconfig import StrictManualConfig
from optexp.models import LeNet5
from optexp.optim import SGD
from optexp.utils import logspace

experiments = [
    Experiment(
        problem=Problem(
            dataset=MNIST(),
            model=LeNet5(),
            batch_size=1000,
            lossfunc=CrossEntropyLoss,
            metrics=[metrics.Accuracy, metrics.CrossEntropyLoss],
        ),
        optim=SGD(lr=lr),
        steps=100,
        eval_every=50,
        hardware_config=StrictManualConfig(
            device="cpu",
        ),
    )
    for lr in logspace(-4, 0, 0)
]

if __name__ == "__main__":
    cli(experiments)
