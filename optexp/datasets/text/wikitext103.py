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
    BASE_URL = "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main"

    def __init__(self, raw: bool):
        self.n = 103
        self.raw = raw
        self.raw_str = "-raw" if raw else ""

    def url(self):
        return f"{self.BASE_URL}/wikitext-{self.n}{self.raw_str}-v1"

    def base_path(self):
        return Config.get_dataset_directory() / (f"WikiText{self.n}" + self.raw_str)

    def txt_file(self, split: str):
        match split:
            case "tr" | "train":
                return self.base_path() / f"wiki{self.raw_str}.train.txt"
            case "va" | "valid":
                return self.base_path() / f"wiki{self.raw_str}.valid.txt"
            case "te" | "test":
                return self.base_path() / f"wiki{self.raw_str}.test.txt"
            case _:
                raise ValueError(
                    f"Unknown split {split}. "
                    f"Expected 'tr', 'train', 'va', 'valid', 'te' or 'test'."
                )

    def all_txt_files(self):
        return [self.txt_file(split) for split in ["tr", "va", "te"]]

    def tokenized_file(self, split):
        return self.base_path() / f"wikitext{self.n}{self.raw_str}_{split}_tokenized.pt"


@frozen
class WikiText103(Dataset, HasClassCounts, Downloadable):

    sequence_length: int = 1024
    raw: bool = True
    tokenizer: Tokenizer = BPETokenizer(vocab_size=50257)

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
        return torch.Size([batch_size, self.sequence_length, self.tokenizer.vocab_size])

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

    def build_tokenizer_if_not_exists(self):
        if not self.tokenizer.has_been_trained(WTFiles(self.raw).base_path()):
            self.tokenizer.build_tokenizer(
                WTFiles(self.raw).base_path(),
                WTFiles(self.raw).txt_file("tr"),
                specials=["<unk>"] if not self.raw else None,
            )

    def get_tokens(self, tr_va: TrVa) -> torch.Tensor:
        self.build_tokenizer_if_not_exists()
        return self.tokenizer.tokenize_and_numify(
            WTFiles(self.raw).base_path(),
            WTFiles(self.raw).txt_file(tr_va),
        )

    def is_downloaded(self) -> bool:
        return all(file.exists() for file in WTFiles(self.raw).all_txt_files())

    def download(self):
        os.makedirs(WTFiles(self.raw).base_path(), exist_ok=True)
        base_url = WTFiles(self.raw).url()
        for file in WTFiles(self.raw).PARQUET_FILES:
            filepath = WTFiles(self.raw).base_path() / Path(file)
            self._download_file(base_url, file, filepath)
            split = file.split("-", maxsplit=1)[0][0:5]
            self._extract(split, filepath)
            filepath.unlink()

    def _extract(self, split, filepath):
        data = pyarrow.parquet.read_table(filepath).to_pydict()
        with open(WTFiles(self.raw).txt_file(split), "a", encoding="utf-8") as f:
            f.write("".join(data["text"]))

    @staticmethod
    def _download_file(base_url, file, filepath):
        with requests.get(f"{base_url}/{file}", stream=True, timeout=10) as r:
            with open(filepath, "wb") as f:
                shutil.copyfileobj(r.raw, f)
