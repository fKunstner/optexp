import torch
from attr import frozen
from numpy.lib.stride_tricks import as_strided, sliding_window_view

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
    allow_overlap: bool = False
    """
    
    args:
        allow_overlap: If True, the sequences can overlap.
            If True, overlapping sequences will be generated, e.g. tokens [1,2,3,4] and sequence 
            length 3 will generate [1,2,3] and [2,3,4]. If False, only non-overlapping sequences,
            e.g. [1,2,3].
    
    The format of the data matrices is as follows:
    sequences_mat: (n_sequences, sequence_len)
    targets_mat: (n_sequences)
    
    all_sequences: If True, the targets are all the tokens in the sequence. Eg;

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
        
    If allow_overlap is True, all sequences (including overlap) are generated, e.g.:
    
        Tokens: [0,1,2,3,4,5]
        
        sequences_mat = [
            [0,1,2],
            [1,2,3],
            [2,3,4],
            [3,4,5],
        ]
        targets_mat = [
            1,
            2,
            3,
            4,
        ]
    
    """

    def generate_subsequences(
        self, tokens: torch.Tensor, sequence_len: int
    ) -> torch.Tensor:
        if self.allow_overlap:
            return tokens.unfold(0, sequence_len, 1)
        else:
            return tokens.unfold(0, sequence_len, sequence_len)

    def tokens_to_sequences_and_targets(
        self, tokens: torch.Tensor, sequence_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequences_mat = self.generate_subsequences(tokens, sequence_len)
        index_of_middle_token = sequence_len // 2
        targets_mat = sequences_mat[:, index_of_middle_token]

        return sequences_mat, targets_mat
