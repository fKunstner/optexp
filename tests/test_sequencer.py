from typing import List

import pytest
import torch

from optexp.datasets.text.wikitext import tokens_to_sequences_and_targets


@pytest.mark.parametrize(
    "tokens,sequence_len,true_sequences,true_targets",
    [
        (
            torch.tensor([1, 2, 3]),
            2,
            [torch.tensor([1, 2])],
            [torch.tensor([2, 3])],
        ),
        (
            torch.tensor([1, 2, 3, 4]),
            2,
            [torch.tensor([1, 2])],
            [torch.tensor([2, 3])],
        ),
        (
            torch.tensor([1, 2, 3, 4, 5]),
            2,
            [torch.tensor([1, 2]), torch.tensor([3, 4])],
            [torch.tensor([2, 3]), torch.tensor([4, 5])],
        ),
    ],
)
def test_sequence(tokens, sequence_len, true_sequences, true_targets):
    sequences_list, targets_list = tokens_to_sequences_and_targets(tokens, sequence_len)
    assert all([torch.equal(a, b) for a, b in zip(sequences_list, true_sequences)])
    assert all([torch.equal(a, b) for a, b in zip(targets_list, true_targets)])


if __name__ == "__main__":
    test_sequence(
        tokens=torch.tensor([1, 2]),
        sequence_len=2,
    )
