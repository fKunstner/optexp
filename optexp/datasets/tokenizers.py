from abc import ABC, abstractmethod
from pathlib import Path

from tqdm import tqdm

from optexp.component import Component
import sentencepiece as sp
import torch


class Tokenizer(ABC, Component):

    @abstractmethod
    def build_tokenizer(self, tokenizer_path: Path, data_path: Path, vocab_size: int):
        raise NotImplementedError()

    @abstractmethod
    def tokenize_and_numify(self, tokenizer_path: Path, data_path: Path):
        raise NotImplementedError()


class BPETokenizer(Tokenizer):

    def build_tokenizer(self, tokenizer_path: Path, data_path: Path, vocab_size: int):
        sp.SentencePieceTrainer.train(
            input=data_path,
            model_prefix=tokenizer_path,
            vocab_size=vocab_size,
            model_type="bpe",
        )

    def tokenize_and_numify(self, tokenizer_path: Path, data_path: Path):
        tokenizer = sp.SentencePieceProcessor()
        success = tokenizer.load(tokenizer_path.absolute().as_posix())
        if not success:
            raise ValueError("Could not load the tokenizer successfully")
        text_lines = open(data_path, "r").readlines()
        tokenized_lines = []
        for line in tqdm(text_lines):
            tokenized_lines.append(
                torch.tensor(tokenizer.encode_as_ids(line), dtype=torch.long)
            )
        return torch.cat(tokenized_lines)


class CharacterTokenizer:
    pass


class WordTokenizers:
    pass
