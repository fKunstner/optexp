from optexp.datasets.image_dataset import MNIST
from optexp.experiments.experiment import Experiment
from optexp.models.vision import LeNet5
from optexp.optimizers import SGD
from optexp.problems import Classification
from optexp.runner.cli import exp_runner_cli
from optexp.runner.slurm.slurm_config import SlurmConfig
from optexp.utils import SEEDS_1, starting_grid_for

dataset = MNIST(batch_size=256)
model = LeNet5()
problem = Classification(model, dataset)
opts_sparse = starting_grid_for([SGD], start=-6, end=-5)

EPOCHS = 2
group = "testing"

experiments = Experiment.generate_experiments_from_opts_and_seeds(
    opts_and_seeds=[(opts_sparse, SEEDS_1)],
    problem=problem,
    epochs=EPOCHS,
    group=group,
)

SLURM_CONFIG = SlurmConfig(hours=10, gb_ram=8, n_cpus=1, n_gpus=1, gpu=True)
if __name__ == "__main__":
    exp_runner_cli(experiments, slurm_config=SLURM_CONFIG)
