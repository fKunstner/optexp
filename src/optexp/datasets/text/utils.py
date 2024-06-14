from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, dataset
from torchtext.data import get_tokenizer
from torchtext.vocab import Vocab, build_vocab_from_iterator


def download_language(save_path: Path, dataset_callable: Callable, tokenizer=None):
    train_iter, val_iter, test_iter = dataset_callable(
        root=save_path.parent, split=("train", "valid", "test")
    )
    if tokenizer is None:
        tokenizer = get_tokenizer("basic_english")

    build_vocab_from_iterator(map(tokenizer, train_iter), specials=["<unk>"])
    build_vocab_from_iterator(map(tokenizer, val_iter), specials=["<unk>"])
    build_vocab_from_iterator(map(tokenizer, test_iter), specials=["<unk>"])


def prepare_loader(train_data, val_data, batch_size, vocab, bptt, merge):
    class_freqs = torch.bincount(train_data)
    train_data = batchify(train_data, batch_size)  # shape ``[seq_len, batch_size]``
    val_data = batchify(val_data, batch_size)
    train_loader = BatchIterator(train_data, bptt, merge)
    val_loader = BatchIterator(val_data, bptt, merge)
    input_shape = (len(vocab),)
    output_shape = (len(vocab) // merge,) if merge else input_shape
    loaders = {"train_loader": train_loader, "val_loader": val_loader}
    return loaders, input_shape, output_shape, class_freqs


def prepare_data_loader(train_data, val_data, batch_size, vocab, bptt):
    class_freqs = torch.bincount(train_data)
    train_dataset = TextData(train_data, bptt)
    val_dataset = TextData(val_data, bptt)

    def collate_fn(batch):
        src_batch, tgt_batch = [], []
        for sample in batch:
            src_batch.append(sample[0])
            tgt_batch.append(sample[1])

        return torch.transpose(torch.vstack(src_batch), 0, 1), torch.transpose(
            torch.vstack(tgt_batch), 0, 1
        ).reshape(-1)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    input_shape = np.array([len(vocab)])
    output_shape = input_shape
    loaders = {"train_loader": train_loader, "val_loader": val_loader}

    return loaders, input_shape, output_shape, class_freqs


def prepare_mixed_size_data_loader(
    train_data, val_data, train_batch_size, eval_batch_size, vocab, bptt
):
    class_freqs = torch.bincount(train_data)
    train_dataset = TextData(train_data, bptt)
    val_dataset = TextData(val_data, bptt)

    def collate_fn(batch):
        src_batch, tgt_batch = [], []
        for sample in batch:
            src_batch.append(sample[0])
            tgt_batch.append(sample[1])

        return torch.transpose(torch.vstack(src_batch), 0, 1), torch.transpose(
            torch.vstack(tgt_batch), 0, 1
        ).reshape(-1)

    train_loader = DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=False, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=train_batch_size, shuffle=False, collate_fn=collate_fn
    )

    eval_train_loader = DataLoader(
        train_dataset, batch_size=eval_batch_size, shuffle=False, collate_fn=collate_fn
    )
    eval_val_loader = DataLoader(
        val_dataset, batch_size=eval_batch_size, shuffle=False, collate_fn=collate_fn
    )

    input_shape = np.array([len(vocab)])
    output_shape = input_shape
    loaders = {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "eval_train_loader": eval_train_loader,
        "eval_val_loader": eval_val_loader,
    }

    return loaders, input_shape, output_shape, class_freqs


def tokenize_and_numify(
    raw_text_iter: dataset.IterableDataset,
    tokenizer: Callable,
    vocab: Vocab,
    cutoff: Optional[float] = None,
):
    data = [
        torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter
    ]

    if cutoff:
        x = int(cutoff * len(data))
        data = data[0:x]

    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


def batchify(data: torch.Tensor, bsz: int) -> torch.Tensor:
    """Divides the data into ``bsz`` separate sequences.

    Removes extra elements that wouldn't cleanly fit.

    Arguments:
        data: Tensor, shape ``[N]``
        bsz: int, batch size

    Returns:
        Tensor of shape ``[N // bsz, bsz]``
    """
    seq_len = data.size(0) // bsz
    data = data[: seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data


class BatchIterator:
    def __init__(
        self, data: torch.Tensor, bptt: int, merge: Optional[int] = None
    ) -> None:
        self.data = data
        self.tgt_len = bptt
        self.merge = merge
        self.i = 0

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i in range(0, self.data.size(0) - 1, self.tgt_len):
            data, targets = self.get_batch(self.i)
            self.i += self.tgt_len
            if len(data) == 0:
                raise StopIteration
            return data, targets
        raise StopIteration

    def get_batch(self, i: int):
        seq_len = min(self.tgt_len, len(self.data) - 1 - i)
        data = self.data[i : i + seq_len]
        targets = self.data[i + 1 : i + 1 + seq_len].reshape(-1)
        targets = (
            torch.floor(targets / self.merge).to(torch.long) if self.merge else targets
        )
        return data, targets


class TextData(torch.utils.data.Dataset):
    def __init__(
        self, data: torch.Tensor, bptt: int, merge: Optional[int] = None
    ) -> None:
        self.data = data
        self.merge = merge
        self.tgt_len = bptt

    def __len__(self):
        return self.data.shape[0] // self.tgt_len

    def __getitem__(self, idx: int):
        seq_len = min(self.tgt_len, len(self.data) - 1 - self.tgt_len * idx)
        data = self.data[idx * self.tgt_len : idx * self.tgt_len + seq_len]
        targets = self.data[
            idx * self.tgt_len + 1 : idx * self.tgt_len + 1 + seq_len
        ].reshape(-1)
        targets = (
            torch.floor(targets / self.merge).to(torch.long) if self.merge else targets
        )
        return data, targets
