import math

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from optexp.runner.exp_state import DataLoaders, ExperimentState, IterationCounter


def test_interaction_counters_are_independent():
    ic1 = IterationCounter()
    ic2 = IterationCounter()

    ic1.start()
    ic1.next_iter()
    ic1.next_iter()
    ic1.next_iter()
    ic1.next_epoch()
    ic1.next_iter()
    ic1.next_iter()
    ic1.next_iter()

    assert ic2.epoch == 0
    assert ic2.step == 0
    assert ic2.step_within_epoch == 0


def test_expstate_with_dataloader_1_batch_per_epoch():
    dataset = TensorDataset(torch.tensor([1]))
    dataloader = DataLoader(dataset, batch_size=1)

    es = ExperimentState(
        1, 1, DataLoaders(dataloader, dataloader, dataloader), 1  # type: ignore
    )

    assert es.iteration_counter.epoch == 0
    assert es.iteration_counter.step == 0
    assert es.iteration_counter.step_within_epoch == 0

    i = 0
    for iter in range(1, 5):
        data = es.get_batch()
        i += 1
        print(es.iteration_counter)
        assert es.iteration_counter.epoch == i
        assert es.iteration_counter.step == i
        assert es.iteration_counter.step_within_epoch == 1

    new_es = ExperimentState(2, 2, 2, 2)  # type: ignore
    assert new_es.iteration_counter.epoch == 0
    assert new_es.iteration_counter.step == 0
    assert new_es.iteration_counter.step_within_epoch == 0


def test_expstate_with_dataloader_3_batch_per_epoch():
    dataset = TensorDataset(torch.tensor([1, 2, 3]))
    dataloader = DataLoader(dataset, batch_size=1)

    es = ExperimentState(
        1, 1, DataLoaders(dataloader, dataloader, dataloader), 1  # type: ignore
    )

    assert es.iteration_counter.epoch == 0
    assert es.iteration_counter.step == 0
    assert es.iteration_counter.step_within_epoch == 0

    i = 0
    for iter in range(1, 10):
        data = es.get_batch()
        i += 1
        assert es.iteration_counter.epoch == math.ceil(i / 3)
        assert es.iteration_counter.step == i
        expected_step_within_epoch = i % 3 if i % 3 != 0 else 3
        assert es.iteration_counter.step_within_epoch == expected_step_within_epoch

    new_es = ExperimentState(2, 2, 2, 2)  # type: ignore
    assert new_es.iteration_counter.epoch == 0
    assert new_es.iteration_counter.step == 0
    assert new_es.iteration_counter.step_within_epoch == 0
