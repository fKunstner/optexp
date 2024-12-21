import os
import shutil
from abc import abstractmethod
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pyarrow.parquet
import requests
import torch
from attrs import frozen
from torch.utils.data import DataLoader

from optexp.component import Component
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


@frozen
class Truncate(Component):
    tr: Optional[int] = None
    va: Optional[int] = None
    te: Optional[int] = None

    def should_truncate(self, split: Split) -> bool:
        return getattr(self, split) is not None

    def truncate_to(self, split: Split) -> Optional[int]:
        return getattr(self, split)


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


@frozen
class WikiTextBase(Dataset, HasClassCounts, Downloadable):

    sequence_length: int = 1024
    raw: bool = False
    tokenizer: Tokenizer = BPETokenizer(vocab_size=50257)
    truncate: Truncate = Truncate()
    task: LanguageTask = PredictNextToken()

    def __post_init__(self):
        if isinstance(self.task, PredictMiddleToken):
            if self.sequence_length % 2 == 0:
                raise ValueError(
                    "When using PredictMiddleToken, sequence_length must be odd."
                )

    def get_dataloader(self, b: int, split: Split, num_workers: int) -> DataLoader:
        return DataLoader(
            self.get_dataset(split),
            batch_size=b,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=True,
        )

    def data_input_shape(self, batch_size: int) -> torch.Size:
        return torch.Size([batch_size, self.sequence_length])

    def model_output_shape(self, batch_size: int) -> torch.Size:
        return torch.Size([batch_size, self.sequence_length, self.tokenizer.vocab_size])

    def get_truncation_information(self) -> Dict[Split, Dict]:
        info = {}
        splits: List[Split] = ["tr", "va", "te"]
        for split in splits:
            x_raw, _ = self.get_data_matrices(split, truncate=False)
            n = x_raw.shape[0]
            info[split] = {
                "truncated to": self.truncate.truncate_to(split),
                "raw": n,
            }
        return info

    @lru_cache()
    def get_num_samples(self, split: Split) -> int:
        sequences, _ = self.get_data_matrices(split)
        return sequences.shape[0]

    @lru_cache()
    def get_num_tokens(self, split: Split) -> int:
        sequences, _ = self.get_data_matrices(split)
        return sequences.numel()

    @lru_cache()
    def class_counts(self, split: Split) -> torch.Tensor:
        _, targets = self.get_data_matrices(split)
        return torch.bincount(targets.view(-1), minlength=self.tokenizer.vocab_size)

    def get_dataset(self, split: Split) -> torch.utils.data.Dataset:
        sequences, targets = self.get_data_matrices(split)
        sequences_list = [sequences[i] for i in range(sequences.shape[0])]
        targets_list = [targets[i] for i in range(sequences.shape[0])]
        return ListDataset(sequences_list, targets_list)

    def build_tokenizer_if_not_exists(self):
        if not self.tokenizer.has_been_trained(self._get_files().base_path()):
            self.tokenizer.build_tokenizer(
                self._get_files().base_path(),
                self._get_files().txt_file("tr"),
                specials=["<unk>"] if not self.raw else None,
            )

    def get_data_matrices(
        self, split: Split, truncate=True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self.task.tokens_to_sequences_and_targets(
            self.get_tokens(split), self.sequence_length
        )

        if truncate and self.truncate.should_truncate(split):
            x = x[: self.truncate.truncate_to(split), :]
            y = y[: self.truncate.truncate_to(split), :]
        return x, y

    def get_tokens(self, split: Split) -> torch.Tensor:
        self.build_tokenizer_if_not_exists()
        tokens = self.tokenizer.tokenize_and_numify(
            self._get_files().base_path(),
            self._get_files().txt_file(split),
        )
        return tokens

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
