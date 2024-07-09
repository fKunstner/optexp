from optexp import Experiment, Problem, cli
from optexp.datasets import MNIST
from optexp.metrics import Accuracy, CrossEntropy
from optexp.models import Linear
from optexp.optim import SGD

experiments = [
    Experiment(
        problem=Problem(
            dataset=MNIST(),
            model=Linear(),
            batch_size=100,
            lossfunc=CrossEntropy(),
            metrics=[
                Accuracy(),
                CrossEntropy(),
            ],
        ),
        group="test",
        optim=SGD(lr=lr),
        steps=2,  # 2 epochs
        eval_every=1,  # Evaluate every epoch
    )
    for lr in [0.1]
]

if __name__ == "__main__":
    cli(experiments)
