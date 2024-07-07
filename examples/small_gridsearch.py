from optexp import Experiment, Problem, cli
from optexp.datasets import MNIST
from optexp.metrics import Accuracy, CrossEntropy
from optexp.models import LeNet5
from optexp.optim import SGD

experiments = [
    Experiment(
        problem=Problem(
            dataset=MNIST(),
            model=LeNet5(),
            batch_size=10000,
            lossfunc=CrossEntropy(),
            metrics=[Accuracy(), CrossEntropy()],
        ),
        optim=SGD(lr=0.01),
        steps=6 * 4,
        eval_every=6,
    )
]

if __name__ == "__main__":
    cli(experiments)
