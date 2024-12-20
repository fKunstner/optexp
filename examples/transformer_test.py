import torch

from optexp import Experiment, Problem, cli
from optexp.datasets import MNIST, WikiText2
from optexp.datasets.text.wikitext import Truncate
from optexp.hardwareconfig import HardwareConfig, StrictManualConfig
from optexp.metrics import Accuracy, CrossEntropy
from optexp.models import LeNet5
from optexp.models.transformer import Transformer
from optexp.optim import SGD, Adam
from optexp.pipes.pipe import SequenceDataPipe

# test = Transformer()
# model = Transformer(n_layers=2, n_head=1, d_model=16, sequence_length=64)

experiments = [
    Experiment(
        problem=Problem(
            dataset=WikiText2(
                sequence_length=1000,
                truncate=Truncate(tr=2000, va=400, te=1000),
            ),
            model=Transformer(n_layers=12, n_head=12, d_model=768),
            batch_size=500,
            lossfunc=CrossEntropy(),
            metrics=[  # Monitor both the accuracy and the loss on the training and validation sets
                Accuracy(),
                CrossEntropy(),
            ],
            datapipe=SequenceDataPipe(),
        ),
        hardware_config=StrictManualConfig(
            micro_batch_size=5, eval_micro_batch_size=50
        ),
        group="testing",
        optim=Adam(lr=lr, weight_decay=0),  # 1e-5),
        steps=10000,  # 2 epochs
        eval_every=1,  # Evaluate every epoch
    )
    for lr in [0.0001]
]

if __name__ == "__main__":
    cli(experiments)
