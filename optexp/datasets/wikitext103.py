import os
import shutil
from pathlib import Path
from typing import List, Tuple, Any

import pyarrow.parquet
import requests
import torch
from attrs import frozen
from torch.utils.data import DataLoader

from optexp.config import Config
from optexp.datasets import Dataset
from optexp.datasets.dataset import TrVa, HasClassCounts, Downloadable
from optexp.datasets.tokenizers import Tokenizer, BPETokenizer
from optexp.datasets.utils import make_list_dataset

DATA_PATH: Path = Config.get_dataset_directory() / "WikiText103"


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

        return make_list_dataset(sequences, targets)

    def get_tokens(self, tr_va: TrVa, vocab_size: int) -> torch.Tensor:
        raw_str = "-raw" if self.raw else ""
        if self.is_tokenized(tr_va):
            return torch.load(DATA_PATH / f"wikitext103{raw_str}_{tr_va}_tokenized.pt")
        text = (
            f"wiki{raw_str}.train.tokens"
            if tr_va == "tr"
            else f"wiki{raw_str}.valid.tokens"
        )
        if not self.has_tokenizer():
            self.tokenizer.build_tokenizer(
                DATA_PATH / f"wikitext103{raw_str}_v={vocab_size}",
                DATA_PATH / f"wiki{raw_str}.train.tokens",
                vocab_size,
            )
        tokens = self.tokenizer.tokenize_and_numify(
            DATA_PATH / f"wikitext103{raw_str}_v={vocab_size}.model",
            DATA_PATH / text,
        )
        torch.save(tokens, DATA_PATH / f"wikitext103{raw_str}_{tr_va}_tokenized.pt")
        return tokens

    def is_tokenized(self, tr_va: TrVa) -> bool:
        raw_str = "-raw" if self.raw else ""
        return (DATA_PATH / f"wikitext103{raw_str}_{tr_va}_tokenized.pt").exists()

    def has_tokenizer(self) -> bool:
        raw_str = "-raw" if self.raw else ""
        return all(
            (DATA_PATH / file).exists()
            for file in [
                f"wikitext103{raw_str}_v={self.vocab_size}.model",
                f"wikitext103{raw_str}_v={self.vocab_size}.vocab",
            ]
        )

    def is_downloaded(self) -> bool:
        raw_str = "-raw" if self.raw else ""
        return all(
            (DATA_PATH / file).exists()
            for file in [
                f"wiki{raw_str}.train.tokens",
                f"wiki{raw_str}.valid.tokens",
                f"wiki{raw_str}.test.tokens",
            ]
        )

    def download(self):
        os.makedirs(DATA_PATH, exist_ok=True)
        raw_str = "-raw" if self.raw else ""
        base_url = f"https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-103{raw_str}-v1"
        files = [
            "validation-00000-of-00001.parquet",
            "test-00000-of-00001.parquet",
            "train-00000-of-00002.parquet",
            "train-00001-of-00002.parquet",
        ]
        for file in files:
            filepath = DATA_PATH / Path(file)
            with requests.get(f"{base_url}/{file}", stream=True) as r:
                with open(filepath, "wb") as f:
                    shutil.copyfileobj(r.raw, f)
            data = pyarrow.parquet.read_table(filepath).to_pydict()
            split = file.split("-", maxsplit=1)[0][0:5]
            with open(
                DATA_PATH / Path(f"wiki{raw_str}.{split}.tokens"),
                "a",
            ) as f:
                f.write("".join(data["text"]))
            filepath.unlink()
