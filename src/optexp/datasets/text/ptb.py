from pathlib import Path
from typing import Optional

from torchtext.data import get_tokenizer
from torchtext.datasets import PennTreebank
from torchtext.vocab import build_vocab_from_iterator

from optexp.datasets import Dataset
from optexp.datasets.text.utils import (
    download_language,
    prepare_loader,
    tokenize_and_numify,
)


class PTB(Dataset):
    pass


def download_ptb(save_path: Path):
    download_language(save_path=save_path, dataset_callable=PennTreebank)


def load_ptb(
    save_path: Path,
    batch_size: str,
    bptt: int,
    device: str,
    merge: Optional[int] = None,
):
    return ptb_loader(
        save_path=save_path,
        batch_size=batch_size,
        bptt=bptt,
        device=device,
        merge=merge,
    )


def ptb_loader(
    save_path: Path,
    batch_size,
    bptt,
    device,
    tiny=False,
    tokenizer=None,
    merge: Optional[int] = None,
):
    train_iter = PennTreebank(root=save_path.parent, split="train")

    if tokenizer is None:
        tokenizer = get_tokenizer("basic_english")

    vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    if tiny:
        train_iter = PennTreebank(root=save_path.parent, split="valid")
        val_iter = PennTreebank(root=save_path.parent, split="test")
    else:
        train_iter, val_iter, _ = PennTreebank(root=save_path.parent)

    train_data = tokenize_and_numify(train_iter, tokenizer, vocab).to(device)
    val_data = tokenize_and_numify(val_iter, tokenizer, vocab).to(device)

    return prepare_loader(train_data, val_data, batch_size, vocab, bptt, merge)
