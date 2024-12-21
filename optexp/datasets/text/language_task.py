import torch
from attr import frozen

from optexp.component import Component


@frozen
class LanguageTask(Component):
    def tokens_to_sequences_and_targets(
        self, tokens: torch.Tensor, sequence_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


@frozen
class PredictNextToken(LanguageTask):
    """
    The format of the data matrices is as follows:
    sequences_mat: (n_sequences, sequence_len)
    targets_mat: (n_sequences, sequence_len)

    For each sequence, the targets are the next token in the sequence. Eg;

        Tokens: [0,1,2,3,4,5]

        sequences_mat = [
            [0,1],
            [2,3],
        ]
        targets_mat = [
            [1,2],
            [3,4],
        ]
    """

    def tokens_to_sequences_and_targets(
        self, tokens: torch.Tensor, sequence_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_tokens = tokens.size()[0]
        n_sequences = n_tokens // sequence_len

        last_target_is_incomplete = n_tokens < (n_sequences * sequence_len) + 1
        if last_target_is_incomplete:
            n_sequences -= 1

        sequences_mat = tokens[0 : n_sequences * sequence_len]
        targets_mat = tokens[1 : n_sequences * sequence_len + 1]

        sequences_mat = sequences_mat.view(n_sequences, sequence_len)
        targets_mat = targets_mat.view(n_sequences, sequence_len)

        return sequences_mat, targets_mat


@frozen
class PredictMiddleToken(LanguageTask):
    """
    The format of the data matrices is as follows:
    sequences_mat: (n_sequences, sequence_len)
    targets_mat: (n_sequences)

    For each sequence, the targets is the token in the middle of the sequence

        Tokens: [0,1,2,3,4,5]

        sequences_mat = [
            [0,1,2],
            [3,4,5],
        ]
        targets_mat = [
            1,
            4,
        ]
    """

    def tokens_to_sequences_and_targets(
        self, tokens: torch.Tensor, sequence_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_tokens = tokens.size()[0]
        n_sequences = n_tokens // sequence_len

        last_target_is_incomplete = n_tokens < (n_sequences * sequence_len) + 1
        if last_target_is_incomplete:
            n_sequences -= 1

        sequences_mat = tokens[0 : n_sequences * sequence_len]
        sequences_mat = sequences_mat.view(n_sequences, sequence_len)

        raise NotImplementedError
