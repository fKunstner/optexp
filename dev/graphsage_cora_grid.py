import optexp
from optexp import Experiment, Problem
from optexp.metrics import Accuracy, CrossEntropy, CrossEntropyPerClass, AccuracyPerClass
from optexp.optim import SGD, Adam
from optexp.runner.slurm.slurm_config import SlurmConfig
from optexp.utils import nice_logspace

from gnnexp.datasets import Cora, Citeseer, PubMed, Computers, Photo
from gnnexp.models import GAT, GraphSAGE

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
            dataset=Cora(),
            batch_size=1,
            lossfunc=CrossEntropy(),
            metrics=[CrossEntropy(), Accuracy(), CrossEntropyPerClass(), AccuracyPerClass()],
            datapipe=optexp.pipes.GraphDataPipe(),
        ),
        group="GraphSAGE_Cora_Gridsearch",
        seed=seed,
        optim=opt,
        eval_every=1,
        steps=1000,
    )
    for opt in optimizers
    for seed in range(5)
]

SLURM_CONFIG = SlurmConfig(hours=1, gb_ram=8, n_cpus=1, n_gpus=1, gpu=True)

if __name__ == "__main__":
    optexp.cli(experiments=exps, slurm_config=SLURM_CONFIG)
