from optexp import Experiment, Problem, cli, use_wandb_config
from optexp.config import use_tqdm_config
from optexp.datasets import WikiText2
from optexp.datasets.text.wikitext import Truncate
from optexp.hardwareconfig import StrictManualConfig
from optexp.metrics import Accuracy, CrossEntropy
from optexp.models.initiliazation import GPT2Initialization
from optexp.models.transformer import Transformer
from optexp.optim import Adam
from optexp.optim.weight_decay_strategies import GPT2WeightDecay
from optexp.pipes.pipe import SequenceDataPipe

experiments = [
    Experiment(
        problem=Problem(
            dataset=WikiText2(
                sequence_length=1024,
                truncate=Truncate(tr=256, va=128, te=0),
            ),
            model=Transformer(
                n_layers=2,
                n_head=12,
                d_model=192,
                d_mlp=None,
                p_residual_dropout=0.1,
                p_attention_dropout=0.1,
                p_embedding_dropout=0.1,
                is_autoregressive=True,
                initialization=GPT2Initialization(),
            ),
            batch_size=32,
            lossfunc=CrossEntropy(),
            metrics=[  # Monitor both the accuracy and the loss on the training and validation sets
                Accuracy(),
                CrossEntropy(),
            ],
            datapipe=SequenceDataPipe(),
        ),
        hardware_config=StrictManualConfig(
            micro_batch_size=4, eval_micro_batch_size=4, num_workers=1
        ),
        group="testing",
        optim=Adam(lr=lr, weight_decay=1e-5, decay_strategy=GPT2WeightDecay()),
        steps=10,  #
        eval_every=2,  # Evaluate every epoch
    )
    for lr in [10.0 ** (-9 / 2)]
]

if __name__ == "__main__":
    with use_wandb_config(enabled=False), use_tqdm_config(enabled=True):
        cli(experiments)
