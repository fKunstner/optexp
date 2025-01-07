from optexp import Experiment, Problem, cli
from optexp.datasets import WikiText2
from optexp.datasets.text.wikitext import Truncate
from optexp.hardwareconfig import StrictManualConfig
from optexp.metrics import Accuracy, CrossEntropy
from optexp.models.transformer import GPT2Small
from optexp.optim import Adam
from optexp.optim.weight_decay_strategies import GPT2WeightDecay
from optexp.pipes.pipe import SequenceDataPipe

experiments = [
    Experiment(
        problem=Problem(
            dataset=WikiText2(
                sequence_length=1024,
                truncate=Truncate(tr=4096, va=256, te=0),
            ),
            model=GPT2Small(),
            batch_size=512,
            lossfunc=CrossEntropy(),
            metrics=[  # Monitor both the accuracy and the loss on the training and validation sets
                Accuracy(),
                CrossEntropy(),
            ],
            datapipe=SequenceDataPipe(),
        ),
        hardware_config=StrictManualConfig(
            micro_batch_size=8, eval_micro_batch_size=32, num_workers=8
        ),
        group="testing",
        optim=Adam(lr=lr, weight_decay=1e-5, decay_strategy=GPT2WeightDecay()),
        steps=10000,  #
        eval_every=8,  # Evaluate every epoch
    )
    for lr in [10.0 ** (-9 / 2)]
]

if __name__ == "__main__":
    cli(experiments)
