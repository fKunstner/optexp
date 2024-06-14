import os
import pickle
from pathlib import Path
from typing import Optional

import torch
from torchtext.data import get_tokenizer
from torchtext.datasets import WikiText103
from torchtext.vocab import build_vocab_from_iterator

from optexp.datasets.text.tokenizers import _get_bpe_tokenizer
from optexp.datasets.text.utils import (
    download_language,
    prepare_data_loader,
    prepare_mixed_size_data_loader,
    tokenize_and_numify,
)


def download_wt103(save_path: Path):
    download_language(save_path=save_path, dataset_callable=WikiText103)


def load_wt103(
    save_path: Path,
    tokenizers_path: Path,
    batch_size: int,
    bptt: int,
    device: str,
    mixed_batch_size: Optional[bool] = False,
    eval_batch_size: Optional[int] = 0,
):
    train_file = save_path / "WikiText103" / "wikitext-103" / "wiki.train.tokens"
    tokenizer = _get_bpe_tokenizer(train_file, tokenizer_save_path=tokenizers_path)
    return wt103_loader(
        save_path,
        batch_size,
        bptt,
        device,
        tokenizer,
        mixed_batch_size=mixed_batch_size,
        eval_batch_size=eval_batch_size,
    )


def wt103_loader(
    save_path: Path,
    batch_size,
    bptt,
    device,
    tokenizer=None,
    mixed_batch_size=False,
    eval_batch_size=0,
):
    train_iter = WikiText103(root=str(save_path.parent), split="train")

    if tokenizer is None:
        tokenizer = get_tokenizer("basic_english")

    vocab_path = save_path / "WikiText103" / "wiki.vocab.pkl"

    if os.path.isfile(vocab_path):
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
    else:
        vocab = build_vocab_from_iterator(
            map(tokenizer, train_iter), specials=["<unk>"]
        )
        vocab.set_default_index(vocab["<unk>"])
        with open(vocab_path, "wb") as f:
            pickle.dump(vocab, f)

    train_iter, val_iter = WikiText103(
        root=str(save_path.parent), split=("train", "valid")  # type: ignore[arg-type]
    )

    train_path = save_path / "WikiText103" / "train.pt"
    val_path = save_path / "WikiText103" / "val.pt"

    if os.path.isfile(train_path):
        train_data = torch.load(train_path)
    else:
        train_data = tokenize_and_numify(train_iter, tokenizer, vocab=vocab)
        torch.save(train_data, train_path)
        train_data = train_data.to(device)

    if os.path.isfile(val_path):
        val_data = torch.load(val_path)
    else:
        val_data = tokenize_and_numify(val_iter, tokenizer, vocab=vocab)
        torch.save(val_data, val_path)
        val_data = val_data.to(device)

    if mixed_batch_size:
        return prepare_mixed_size_data_loader(
            train_data, val_data, batch_size, eval_batch_size, vocab, bptt
        )
    return prepare_data_loader(train_data, val_data, batch_size, vocab, bptt)
