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
            batch_size=10000,
            lossfunc=CrossEntropyLoss,
            metrics=[metrics.Accuracy, metrics.CrossEntropyLoss],
        ),
        optim=SGD(lr=0.01),
        steps=6 * 4,
        eval_every=6,
    )
]

if __name__ == "__main__":
    cli(experiments)
