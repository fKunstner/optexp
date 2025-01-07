from typing import Callable

import numpy as np
import pytest
import torch
from attr import frozen
from torch.utils.data import DataLoader, TensorDataset

from optexp import Experiment, Problem, use_wandb_config
from optexp.config import use_tqdm_config
from optexp.datasets import Dataset
from optexp.datasets.dataset import Split
from optexp.datasets.utils import make_dataloader
from optexp.hardwareconfig import StrictManualConfig
from optexp.metrics import MSE, Accuracy
from optexp.metrics.metrics import MAE
from optexp.models import Linear
from optexp.optim import SGD
from optexp.pipes import TensorDataPipe
from optexp.runner.exp_state import ExperimentState
from optexp.runner.init_callback import InitCallback
from optexp.runner.runner import run_experiment


@frozen
class DummyDataset(Dataset):

    N = 100

    def get_dataloader(self, b: int, split: Split, num_workers: int) -> DataLoader:
        X, y = torch.ones(self.N, 1), torch.zeros(self.N, 1)
        return make_dataloader(TensorDataset(X, y), b, num_workers)

    def data_input_shape(self, batch_size: int) -> torch.Size:
        return torch.Size([batch_size, 1])

    def model_output_shape(self, batch_size: int) -> torch.Size:
        return torch.Size([batch_size, 1])

    def get_num_samples(self, split: Split) -> int:
        return 100

    def has_test_set(self) -> bool:
        return False


@frozen
class CallbackZeroInit(InitCallback):
    def __call__(
        self,
        exp: Experiment,
        exp_state: ExperimentState,
        log: Callable[[str], None],
    ) -> ExperimentState:
        for p in exp_state.model.parameters():
            p.data.fill_(1.0)
        return exp_state


def compute_progress(microbs, bs, steps, lr):
    exp = Experiment(
        optim=SGD(lr=lr),
        problem=Problem(
            dataset=DummyDataset(),
            model=Linear(bias=False),
            lossfunc=MAE(),
            metrics=[MAE()],
            batch_size=bs,
            datapipe=TensorDataPipe(),
            init_callback=CallbackZeroInit(),
        ),
        group="test",
        eval_every=1,
        seed=0,
        steps=steps,
        hardware_config=StrictManualConfig(
            num_devices=1,
            micro_batch_size=microbs,
            eval_micro_batch_size=microbs,
            num_workers=0,
            device="cpu",
        ),
    )

    with use_wandb_config(enabled=False), use_tqdm_config(enabled=True):
        exp_state = run_experiment(exp)
        return list(exp_state.model.parameters())[0].item()


@pytest.mark.parametrize(
    "microbs,bs,steps,lr",
    [
        (100, 100, 1, 0.1),
        (10, 100, 1, 0.1),
        (1, 10, 1, 0.1),
        (100, 100, 3, 0.1),
        (10, 100, 3, 0.1),
        (1, 10, 3, 0.1),
    ],
)
def test_sequence(microbs, bs, steps, lr):
    expected_error = 1 - steps * lr
    if steps * lr > 1:
        raise ValueError("Invalid test case")

    assert np.allclose(compute_progress(microbs, bs, steps, lr), expected_error)


if __name__ == "__main__":

    print(compute_progress(1, 10, 1, 0.1))
