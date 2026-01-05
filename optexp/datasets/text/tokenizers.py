from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

import torch
from attr import frozen
from tokenizers import Tokenizer as HF_Tokenizer
from tokenizers.implementations import ByteLevelBPETokenizer
from tqdm import tqdm

from optexp.component import Component

TOKENIZER_FILE = "tokenizer.json"
VOCAB_FILE = "vocab.json"
MERGE_FILE = "merges.txt"


@frozen
class Tokenizer(ABC, Component):

    vocab_size: int

    @abstractmethod
    def build_tokenizer(
        self,
        base_path: Path,
        data_path: Path,
        specials: Optional[List[str]] = None,
    ):
        raise NotImplementedError()

    @abstractmethod
    def tokenize_and_numify(self, base_path: Path, data_path: Path):
        raise NotImplementedError()

    @abstractmethod
    def has_been_trained(self, base_path: Path):
        raise NotImplementedError()


@frozen
class BPETokenizer(Tokenizer):

    vocab_size: int

    def build_tokenizer(
        self,
        base_path: Path,
        data_path: Path,
        specials: Optional[List[str]] = None,
    ):
        tokenizer = ByteLevelBPETokenizer(add_prefix_space=True)
        tokenizer.train(
            str(data_path),
            vocab_size=self.vocab_size,
            min_frequency=2,
            show_progress=True,
            special_tokens=specials if specials else [],
        )
        tokenizer.save_model(str(self._tokenizer_path(base_path)))
        tokenizer.save(str(self._tokenizer_path(base_path) / TOKENIZER_FILE))

    def vocabulary(self, base_path: Path):
        tokenizer = HF_Tokenizer.from_file(
            str(self._tokenizer_path(base_path) / TOKENIZER_FILE)
        )
        return tokenizer.get_vocab()

    def tokenize_and_numify(self, base_path: Path, data_path: Path):
        if self.tokenized_path(base_path, data_path).exists():
            return torch.load(self.tokenized_path(base_path, data_path))

        tokenizer = HF_Tokenizer.from_file(
            str(self._tokenizer_path(base_path) / TOKENIZER_FILE)
        )

        with open(data_path, "r", encoding="utf-8") as f:
            text_lines = f.readlines()
            tokenized_lines = []
            for line in tqdm(text_lines):
                tokenized_lines.append(
                    torch.tensor(tokenizer.encode(line).ids, dtype=torch.long)
                )

        tokens = torch.cat(tokenized_lines)
        torch.save(tokens, self.tokenized_path(base_path, data_path))
        return tokens

    def has_been_trained(self, base_path: Path):
        return all(
            file.exists()
            for file in [
                self._tokenizer_path(base_path) / MERGE_FILE,
                self._tokenizer_path(base_path) / VOCAB_FILE,
            ]
        )

    def _tokenizer_path(self, base_path: Path):
        base_path = base_path / self.equivalent_definition()
        base_path.mkdir(parents=True, exist_ok=True)
        return base_path

    def tokenized_path(self, dataset_path, file_path) -> Path:
        return self._tokenizer_path(dataset_path) / (file_path.name + ".tokenized")
