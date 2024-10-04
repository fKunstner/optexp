import os
import shutil
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pyarrow.parquet
import requests
import torch
from attrs import frozen
from torch.utils.data import DataLoader

from optexp.config import Config
from optexp.datasets.dataset import Dataset, Downloadable, HasClassCounts, Split
from optexp.datasets.text.tokenizers import BPETokenizer, Tokenizer
from optexp.datasets.utils import ListDataset


class WTFiles:
    _TXT_FILES: Dict[Split, str] = {
        "te": "wiki.test",
        "tr": "wiki.train",
        "va": "wiki.valid",
    }
    _BASE_URL = "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main"
    _URLS_103: Dict[Split, List[str]] = {
        "te": ["test-00000-of-00001.parquet"],
        "tr": ["train-00000-of-00002.parquet", "train-00001-of-00002.parquet"],
        "va": ["validation-00000-of-00001.parquet"],
    }
    _URLS_2: Dict[Split, List[str]] = {
        "te": ["test-00000-of-00001.parquet"],
        "tr": ["train-00000-of-00001.parquet"],
        "va": ["validation-00000-of-00001.parquet"],
    }

    def __init__(self, n: int, raw: bool):

        if n == 2:
            self.parquet_files = self._URLS_2
        elif n == 103:
            self.parquet_files = self._URLS_103
        else:
            raise ValueError(f"WikiText{n} unknown. {n} must be 2 or 103")

        self.n = n
        self.raw = raw
        self.raw_str = "-raw" if raw else ""

    def url(self):
        return f"{self._BASE_URL}/wikitext-{self.n}{self.raw_str}-v1"

    def base_path(self):
        return Config.get_dataset_directory() / (f"WikiText{self.n}" + self.raw_str)

    def txt_file(self, split: Split):
        return self.base_path() / f"{self._TXT_FILES[split]}.txt"

    def all_txt_files(self):
        return [self.txt_file(split) for split in self._TXT_FILES]

    def tokenized_file(self, split):
        return self.base_path() / f"wikitext{self.n}{self.raw_str}_{split}_tokenized.pt"


def collate_fn(batch: List[Tuple[Any, Any]]) -> Tuple[Any, Any]:
    src_batch, tgt_batch = [], []
    for sample in batch:
        src_batch.append(sample[0])
        tgt_batch.append(sample[1])

    return torch.vstack(src_batch), torch.vstack(tgt_batch).reshape(-1)


@frozen
class WikiTextBase(Dataset, HasClassCounts, Downloadable):

    sequence_length: int = 1024
    raw: bool = False
    tokenizer: Tokenizer = BPETokenizer(vocab_size=50257)

    def get_dataloader(self, b: int, split: Split, num_workers: int) -> DataLoader:
        dataset = self.get_dataset(split)

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

    def get_num_samples(self, split: Split) -> int:
        tokens = self.get_tokens(split)
        n_sequences = tokens.size()[0] // self.sequence_length
        return n_sequences * self.sequence_length

    def class_counts(self, split: Split) -> torch.Tensor:
        tokens = self.get_tokens(split)
        n_sequences = tokens.size()[0] // self.sequence_length
        cut_tokens = tokens[0 : n_sequences * self.sequence_length]
        return torch.bincount(cut_tokens)

    def get_dataset(self, split: Split) -> torch.utils.data.Dataset:
        tokens = self.get_tokens(split)
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
        if not self.tokenizer.has_been_trained(self._get_files().base_path()):
            self.tokenizer.build_tokenizer(
                self._get_files().base_path(),
                self._get_files().txt_file("tr"),
                specials=["<unk>"] if not self.raw else None,
            )

    def get_tokens(self, split: Split) -> torch.Tensor:
        self.build_tokenizer_if_not_exists()
        return self.tokenizer.tokenize_and_numify(
            self._get_files().base_path(),
            self._get_files().txt_file(split),
        )

    def is_downloaded(self) -> bool:
        return all(file.exists() for file in self._get_files().all_txt_files())

    def download(self):
        os.makedirs(self._get_files().base_path(), exist_ok=True)
        base_url = self._get_files().url()
        for split in self._get_files().parquet_files.keys():
            parquet_files = self._get_files().parquet_files[split]
            txt_file = self._get_files().txt_file(split)

            if txt_file.exists():
                txt_file.unlink()

            for parquet_filename in parquet_files:
                parquet_path = self._get_files().base_path() / Path(parquet_filename)
                self._download_file(f"{base_url}/{parquet_filename}", parquet_path)
                self._extract_append_to(parquet_path, txt_file)
                parquet_path.unlink()

    @staticmethod
    def _extract_append_to(parquet_file, final_txt_tile):
        data = pyarrow.parquet.read_table(parquet_file).to_pydict()
        with open(final_txt_tile, "a", encoding="utf-8") as f:
            f.write("".join(data["text"]))

    @staticmethod
    def _download_file(url, filepath):
        with requests.get(url, stream=True, timeout=10) as r:
            with open(filepath, "wb") as f:
                shutil.copyfileobj(r.raw, f)

    @abstractmethod
    def _get_files(self) -> WTFiles:
        raise NotImplementedError

    def has_test_set(self) -> bool:
        return False


class WikiText2(WikiTextBase):

    def _get_files(self) -> WTFiles:
        return WTFiles(n=2, raw=self.raw)


class WikiText103(WikiTextBase):

    def _get_files(self) -> WTFiles:
        return WTFiles(n=103, raw=self.raw)
