from optexp import Experiment, Problem, cli
from optexp.datasets import MNIST, WikiText2
from optexp.metrics import Accuracy, CrossEntropy
from optexp.models import LeNet5
from optexp.models.transformer import Transformer
from optexp.optim import SGD

# test = Transformer()
# model = Transformer(n_layers=2, n_head=1, d_model=16, sequence_length=64)

experiments = [
    Experiment(
        problem=Problem(
            dataset=WikiText2(),
            model=Transformer(n_layers=2, n_head=1, d_model=16, sequence_length=64),
            batch_size=100,
            lossfunc=CrossEntropy(),
            metrics=[  # Monitor both the accuracy and the loss on the training and validation sets
                Accuracy(),
                CrossEntropy(),
            ],
        ),
        group="testing",
        optim=SGD(lr=lr),
        steps=10,  # 2 epochs
        eval_every=5,  # Evaluate every epoch
    )
    for lr in [0.1, 0.01, 0.001]
]

if __name__ == "__main__":
    cli(experiments)
