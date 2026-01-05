import time
from pprint import pprint

from gnnexp.datasets import Citeseer, Computers, Cora, Photo, PubMed
from gnnexp.models import GAT, GraphSAGE
from tqdm import tqdm

import optexp
from optexp import Experiment, Problem
from optexp.metrics import (
    Accuracy,
    AccuracyPerClass,
    CrossEntropy,
    CrossEntropyPerClass,
)
from optexp.optim import SGD, Adam
from optexp.results.wandb_api import WandbAPI, get_wandb_runs_by_hash
from optexp.results.wandb_data_logger import remove_experiments_that_are_already_saved
from optexp.runner.slurm.slurm_config import SlurmConfig
from optexp.utils import nice_logspace

lrs = nice_logspace(start=-5, end=4, base=10, density=0)
wds = nice_logspace(start=-5, end=4, base=10, density=0)
optimizers = [
    *[Adam(lr=lr, weight_decay=wd) for lr in lrs for wd in wds],
    *[SGD(lr=lr, weight_decay=wd) for lr in lrs for wd in wds],
]


exps = [
    Experiment(
        problem=Problem(
            model=GraphSAGE(),
            dataset=Photo(),
            batch_size=1,
            lossfunc=CrossEntropy(),
            metrics=[
                CrossEntropy(),
                Accuracy(),
                CrossEntropyPerClass(),
                AccuracyPerClass(),
            ],
            datapipe=optexp.pipes.GraphDataPipe(),
        ),
        group="GraphSAGE_Photo_Gridsearch",
        seed=seed,
        optim=opt,
        eval_every=1,
        steps=1000,
    )
    for opt in optimizers
    for seed in range(5)
]

SLURM_CONFIG = SlurmConfig(
    hours=2, gb_ram=8, n_cpus=1, n_gpus=1, gpu=True  # , jobs_per_node=10
)

if __name__ == "__main__":
    optexp.cli(experiments=exps, slurm_config=SLURM_CONFIG)
