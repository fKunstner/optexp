import os
import shutil
from pathlib import Path
from typing import Any, List, Tuple

import pyarrow.parquet
import requests
import torch
from attrs import frozen
from torch.utils.data import DataLoader

from optexp.config import Config
from optexp.datasets.dataset import Dataset, Downloadable, HasClassCounts, TrVa
from optexp.datasets.text.tokenizers import BPETokenizer, Tokenizer
from optexp.datasets.utils import ListDataset


class WTFiles:
    PARQUET_FILES = [
        "validation-00000-of-00001.parquet",
        "test-00000-of-00001.parquet",
        "train-00000-of-00002.parquet",
        "train-00001-of-00002.parquet",
    ]
    DATA_PATH: Path = Config.get_dataset_directory() / "WikiText103"
    BASE_URL = "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main"

    def __init__(self, raw: bool):
        self.raw = raw
        self.raw_str = "-raw" if raw else ""

    def full_path(self, file: str):
        return self.DATA_PATH / file

    def tokens_file(self, split: str):
        match split:
            case "tr" | "train":
                return self.full_path(f"wiki{self.raw_str}.train.tokens")
            case "va" | "valid":
                return self.full_path(f"wiki{self.raw_str}.valid.tokens")
            case "te" | "test":
                return self.full_path(f"wiki{self.raw_str}.test.tokens")
            case _:
                raise ValueError(
                    f"Unknown split {split}. "
                    f"Expected 'tr', 'train', 'va', 'valid', 'te' or 'test'."
                )

    def all_tokens_file(self):
        return [self.tokens_file(split) for split in ["tr", "va", "te"]]

    def tokenizer_base_path(self, vocab_size: int):
        return self.full_path(f"wikitext103{self.raw_str}_v={vocab_size}")

    def tokenized_file(self, tr_va):
        return self.full_path(f"wikitext103{self.raw_str}_{tr_va}_tokenized.pt")

    def merge_file(self, vocab_size: int):
        return self.full_path(f"wikitext103{self.raw_str}_v={vocab_size}-merges.txt")

    def vocab_file(self, vocab_size: int):
        return self.full_path(f"wikitext103{self.raw_str}_v={vocab_size}-vocab.txt")

    def dataset_url(self):
        return f"{self.BASE_URL}/wikitext-103{self.raw_str}-v1"


@frozen
class WikiText103(Dataset, HasClassCounts, Downloadable):

    sequence_length: int = 1024
    vocab_size: int = 50257
    raw: bool = True
    tokenizer: Tokenizer = BPETokenizer()

    def get_dataloader(self, b: int, tr_va: TrVa, num_workers: int) -> DataLoader:
        dataset = self.get_dataset(tr_va)

        def collate_fn(batch: List[Tuple[Any, Any]]) -> Tuple[Any, Any]:
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
        tokens = self.get_tokens(tr_va)
        n_sequences = tokens.size()[0] // self.sequence_length
        return n_sequences * self.sequence_length

    def class_counts(self, tr_va: TrVa) -> torch.Tensor:
        tokens = self.get_tokens(tr_va)
        n_sequences = tokens.size()[0] // self.sequence_length
        cut_tokens = tokens[0 : n_sequences * self.sequence_length]
        return torch.bincount(cut_tokens)

    def get_dataset(self, tr_va: TrVa) -> torch.utils.data.Dataset:
        tokens = self.get_tokens(tr_va)
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

        return ListDataset(sequences, targets)

    def get_tokens(self, tr_va: TrVa) -> torch.Tensor:
        if self.is_tokenized(tr_va):
            return torch.load(WTFiles(self.raw).tokenized_file(tr_va))

        if not self.has_tokenizer():
            self.tokenizer.build_tokenizer(
                WTFiles(self.raw).tokenizer_base_path(self.vocab_size),
                WTFiles(self.raw).tokens_file("tr"),
                self.vocab_size,
                specials=["<unk>"] if not self.raw else None,
            )

        tokens = self.tokenizer.tokenize_and_numify(
            WTFiles(self.raw).tokenizer_base_path(self.vocab_size),
            WTFiles(self.raw).tokens_file(tr_va),
        )
        torch.save(tokens, WTFiles(self.raw).tokenized_file(tr_va))
        return tokens

    def is_tokenized(self, tr_va: TrVa) -> bool:
        return (WTFiles(self.raw).tokenized_file(tr_va)).exists()

    def has_tokenizer(self) -> bool:
        return all(
            file.exists()
            for file in [
                WTFiles(self.raw).vocab_file(self.vocab_size),
                WTFiles(self.raw).merge_file(self.vocab_size),
            ]
        )

    def is_downloaded(self) -> bool:
        return all(file.exists() for file in WTFiles(self.raw).all_tokens_file())

    def download(self):
        os.makedirs(WTFiles(self.raw).DATA_PATH, exist_ok=True)
        base_url = WTFiles(self.raw).dataset_url()
        files = WTFiles(self.raw).PARQUET_FILES
        for file in files:
            filepath = WTFiles(self.raw).DATA_PATH / Path(file)
            self._download_file(base_url, file, filepath)
            split = file.split("-", maxsplit=1)[0][0:5]
            self._extract(split, filepath)
            filepath.unlink()

    def _extract(self, split, filepath):
        data = pyarrow.parquet.read_table(filepath).to_pydict()
        with open(WTFiles(self.raw).tokens_file(split), "a", encoding="utf-8") as f:
            f.write("".join(data["text"]))

    @staticmethod
    def _download_file(base_url, file, filepath):
        with requests.get(f"{base_url}/{file}", stream=True, timeout=10) as r:
            with open(filepath, "wb", encoding="utf-8") as f:
                shutil.copyfileobj(r.raw, f)
