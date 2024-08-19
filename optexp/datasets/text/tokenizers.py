from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

import torch
from tokenizers.implementations import ByteLevelBPETokenizer
from tqdm import tqdm

from optexp.component import Component


class Tokenizer(ABC, Component):

    @abstractmethod
    def build_tokenizer(
        self,
        tokenizer_path: Path,
        data_path: Path,
        vocab_size: int,
        specials: Optional[List[str]] = None,
    ):
        raise NotImplementedError()

    @abstractmethod
    def tokenize_and_numify(self, dataset_path: Path, data_path: Path, vocab_size: int):
        raise NotImplementedError()

    @abstractmethod
    def has_been_trained(self, dataset_path: Path, vocab_size: int):
        raise NotImplementedError()


class BPETokenizer(Tokenizer):

    def build_tokenizer(
        self,
        tokenizer_path: Path,
        data_path: Path,
        vocab_size: int,
        specials: Optional[List[str]] = None,
    ):
        tokenizer = ByteLevelBPETokenizer(add_prefix_space=True)
        tokenizer.train(
            data_path.absolute().as_posix(),
            vocab_size=vocab_size,
            min_frequency=2,
            show_progress=True,
            special_tokens=specials if specials else [],
        )
        tokenizer.save_model(
            tokenizer_path.parents[0].absolute().as_posix(), tokenizer_path.name
        )

    def tokenize_and_numify(self, dataset_path: Path, file_path: Path, vocab_size: int):
        if self.tokenized_path(dataset_path, file_path).exists():
            return torch.load(self.tokenized_path(dataset_path, file_path))

        tokenizer = ByteLevelBPETokenizer(
            str(self._tokenizer_path(dataset_path) / self._merge_file(vocab_size)),
            str(self._tokenizer_path(dataset_path) / self._vocab_file(vocab_size)),
            add_prefix_space=True,
        )

        with open(file_path, "r", encoding="utf-8") as f:
            text_lines = f.readlines()
            tokenized_lines = []
            for line in tqdm(text_lines):
                tokenized_lines.append(
                    torch.tensor(tokenizer.encode(line).ids, dtype=torch.long)
                )

        tokens = torch.cat(tokenized_lines)
        torch.save(tokens, self.tokenized_path(dataset_path, file_path))
        return tokens

    def has_been_trained(self, dataset_path: Path, vocab_size: int):
        return all(
            file.exists()
            for file in [
                self._tokenizer_path(dataset_path) / self._merge_file(vocab_size),
                self._tokenizer_path(dataset_path) / self._vocab_file(vocab_size),
            ]
        )

    @staticmethod
    def _merge_file(vocab_size: int):
        return f"merges-v={vocab_size}.txt"

    @staticmethod
    def _vocab_file(vocab_size: int):
        return f"vocab-v={vocab_size}.txt"

    def _tokenizer_path(self, base_path: Path):
        return base_path / self.equivalent_definition()

    def tokenized_path(self, dataset_path, file_path) -> Path:
        return self._tokenizer_path(dataset_path) / (file_path.name + ".tokenized")
