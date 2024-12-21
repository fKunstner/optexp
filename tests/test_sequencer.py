from functools import partial
from typing import List

import pytest
import torch

from optexp.datasets.text.language_task import PredictMiddleToken, PredictNextToken


@pytest.mark.parametrize(
    "task,tokens,sequence_len,true_sequences,true_targets",
    [
        (
            PredictNextToken,
            torch.tensor([1, 2, 3]),
            2,
            [torch.tensor([1, 2])],
            [torch.tensor([2, 3])],
        ),
        (
            PredictNextToken,
            torch.tensor([1, 2, 3, 4]),
            2,
            [torch.tensor([1, 2])],
            [torch.tensor([2, 3])],
        ),
        (
            PredictNextToken,
            torch.tensor([1, 2, 3, 4, 5]),
            2,
            [torch.tensor([1, 2]), torch.tensor([3, 4])],
            [torch.tensor([2, 3]), torch.tensor([4, 5])],
        ),
        (
            PredictMiddleToken,
            torch.tensor([1, 2, 3, 4, 5, 6]),
            3,
            torch.tensor(
                [
                    [1, 2, 3],
                    [4, 5, 6],
                ]
            ),
            torch.tensor([2, 5]),
        ),
        (
            partial(PredictMiddleToken, allow_overlap=True),
            torch.tensor([1, 2, 3, 4, 5, 6]),
            3,
            torch.tensor(
                [
                    [1, 2, 3],
                    [2, 3, 4],
                    [3, 4, 5],
                    [4, 5, 6],
                ]
            ),
            torch.tensor([2, 3, 4, 5]),
        ),
    ],
)
def test_sequence(task, tokens, sequence_len, true_sequences, true_targets):
    sequences_list, targets_list = task().tokens_to_sequences_and_targets(
        tokens, sequence_len
    )
    assert all([torch.equal(a, b) for a, b in zip(sequences_list, true_sequences)])
    assert all([torch.equal(a, b) for a, b in zip(targets_list, true_targets)])


if __name__ == "__main__":

    test_sequence(
        PredictMiddleToken,
        torch.tensor([1, 2, 3, 4, 5, 6]),
        3,
        [torch.tensor([1, 2, 3]), torch.tensor([2])],
        [torch.tensor([4, 5, 6]), torch.tensor([5])],
    )
