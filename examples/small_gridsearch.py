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
            batch_size=100,
            lossfunc=CrossEntropy(),
            metrics=[  # Monitor both the accuracy and the loss on the training and validation sets
                Accuracy(),
                CrossEntropy(),
            ],
        ),
        group="first_gridsearch",
        optim=SGD(lr=lr),
        steps=10,  # 2 epochs
        eval_every=5,  # Evaluate every epoch
    )
    for lr in [0.1, 0.01, 0.001]
]

if __name__ == "__main__":
    cli(experiments)
