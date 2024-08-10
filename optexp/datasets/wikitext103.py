import os
import shutil
from pathlib import Path

import pyarrow.parquet
import requests
import torch
from attrs import frozen
from torch.utils.data import DataLoader

from optexp.config import Config
from optexp.datasets import Dataset
from optexp.datasets.dataset import TrVa, HasClassCounts, Downloadable
from optexp.datasets.tokenizers import Tokenizer, BPETokenizer


#  TODO <unk> considered harmful
@frozen
class WikiText103(Dataset, HasClassCounts, Downloadable):

    sequence_length: int = 1024
    vocab_size: int = 50257
    tokenizer: Tokenizer = BPETokenizer()
    directory: Path = Config.get_dataset_directory() / "WikiText103"

    def get_dataloader(self, b: int, tr_va: TrVa, num_workers: int) -> DataLoader:
        dataset = self.get_dataset(tr_va)

        def collate_fn(batch):
            src_batch, tgt_batch = [], []
            for sample in batch:
                src_batch.append(sample[0])
                tgt_batch.append(sample[1])

            return torch.vstack(src_batch), torch.vstack(tgt_batch).reshape(-1)

        loader = DataLoader(
            dataset,
            batch_size=b,
            collate_fn=collate_fn,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=True,
        )
        return loader

    def input_shape(self, batch_size: int) -> torch.Size:
        return torch.Size([batch_size, self.sequence_length, 1])

    def output_shape(self, batch_size: int) -> torch.Size:
        return torch.Size([batch_size, self.sequence_length, self.vocab_size])

    def get_num_samples(self, tr_va: TrVa) -> int:
        tokens = self.get_tokens(tr_va, self.vocab_size)
        n_sequences = tokens.size()[0] // self.sequence_length
        return n_sequences * self.sequence_length

    def class_counts(self, tr_va: TrVa) -> torch.Tensor:
        tokens = self.get_tokens(tr_va, self.vocab_size)
        n_sequences = tokens.size()[0] // self.sequence_length
        cut_tokens = tokens[0 : n_sequences * self.sequence_length]
        return torch.bincount(cut_tokens)

    def get_dataset(self, tr_va: TrVa) -> torch.utils.data.Dataset:
        tokens = self.get_tokens(tr_va, self.vocab_size)
        n_tokens = tokens.size()[0]
        n_sequences = n_tokens // self.sequence_length
        cut_sequences = tokens[0 : n_sequences * self.sequence_length].view(
            n_sequences, self.sequence_length
        )
        sequences = []
        targets = []
        for i in range(n_sequences):
            sequence = cut_sequences[i]
            target = tokens[
                i * self.sequence_length + 1 : (i + 1) * self.sequence_length + 1
            ]
            sequences.append(sequence)
            targets.append(target)

        class TorchDataset(torch.utils.data.Dataset):
            def __init__(self):
                self.sequences = sequences
                self.targets = targets

            def __len__(self):
                return len(self.sequences)

            def __getitem__(self, idx: int):
                return self.sequences[idx], self.targets[idx]

        return TorchDataset()

    def get_tokens(self, tr_va: TrVa, vocab_size: int):
        if self.is_tokenized(tr_va):
            tokens = torch.load(
                open(
                    self.directory / f"wikitext103_{tr_va}_tokenized.pt",
                    "rb",
                )
            )
            return tokens
        raw_data = "wiki.train.tokens" if tr_va == "tr" else "wiki.valid.tokens"
        if not self.has_tokenizer():
            self.tokenizer.build_tokenizer(
                self.directory / f"wikitext103_v={vocab_size}",
                self.directory / "wiki.train.tokens",
                vocab_size,
            )
        tokens = self.tokenizer.tokenize_and_numify(
            self.directory / f"wikitext103_v={vocab_size}.model",
            self.directory / raw_data,
        )
        torch.save(
            tokens,
            open(
                self.directory / f"wikitext103_{tr_va}_tokenized.pt",
                "wb",
            ),
        )
        return tokens

    def is_tokenized(self, tr_va: TrVa):
        return (self.directory / f"wikitext103_{tr_va}_tokenized.pt").exists()

    def has_tokenizer(self):
        return all(
            (self.directory / file).exists()
            for file in [
                f"wikitext103_v={self.vocab_size}.model",
                f"wikitext103_v={self.vocab_size}.vocab",
            ]
        )

    def is_downloaded(self):
        return all(
            (self.directory / file).exists()
            for file in [
                "wiki.train.tokens",
                "wiki.valid.tokens",
                "wiki.test.tokens",
            ]
        )

    def download(self):
        os.makedirs(self.directory, exist_ok=True)
        base_url = "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-103-v1"
        files = [
            "validation-00000-of-00001.parquet",
            "test-00000-of-00001.parquet",
            "train-00000-of-00002.parquet",
            "train-00001-of-00002.parquet",
        ]
        for file in files:
            filepath = self.directory / Path(file)
            with requests.get(f"{base_url}/{file}", stream=True) as r:
                with open(filepath, "wb") as f:
                    shutil.copyfileobj(r.raw, f)
            data = pyarrow.parquet.read_table(filepath).to_pydict()
            if file.startswith("train"):
                split = "train"
            if file.startswith("validation"):
                split = "valid"
            if file.startswith("test"):
                split = "test"
            with open(
                self.directory / Path(f"wiki.{split}.tokens"),
                "a",
            ) as f:
                f.write("".join(data["text"]))
            filepath.unlink()
