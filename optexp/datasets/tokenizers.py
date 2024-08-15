from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List
from tokenizers.implementations import ByteLevelBPETokenizer
from tqdm import tqdm
import torch

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
    def tokenize_and_numify(self, tokenizer_path: Path, data_path: Path):
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

    def tokenize_and_numify(self, tokenizer_path: Path, data_path: Path):
        tokenizer = ByteLevelBPETokenizer(
            f"{tokenizer_path}-vocab.json",
            f"{tokenizer_path}-merges.txt",
            add_prefix_space=True,
        )

        with open(data_path, "r", encoding="utf-8") as f:
            text_lines = f.readlines()
            tokenized_lines = []
            for line in tqdm(text_lines):
                tokenized_lines.append(
                    torch.tensor(tokenizer.encode(line).ids, dtype=torch.long)
                )
        return torch.cat(tokenized_lines)


class CharacterTokenizer:
    pass


class WordTokenizers:
    pass
