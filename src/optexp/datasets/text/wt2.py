from pathlib import Path

from torchtext.data import get_tokenizer
from torchtext.datasets import WikiText2
from torchtext.vocab import build_vocab_from_iterator

from optexp.datasets.text.utils import (
    download_language,
    prepare_data_loader,
    tokenize_and_numify,
)


def download_wt2(save_path: Path):
    download_language(save_path=save_path, dataset_callable=WikiText2)


def load_wt2(
    save_path: Path,
    batch_size: str,
    bptt: int,
    device: str,
):
    return wt2_loader(
        save_path=save_path,
        batch_size=batch_size,
        bptt=bptt,
        device=device,
    )


def wt2_loader(
    save_path: Path,
    batch_size,
    bptt,
    device,
    tokenizer=None,
):
    train_iter = WikiText2(root=str(save_path.parent), split="train")

    if tokenizer is None:
        tokenizer = get_tokenizer("basic_english")

    vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    train_iter, val_iter, _ = WikiText2(
        root=str(save_path.parent), split=("train", "valid")  # type: ignore[arg-type]
    )

    train_data = tokenize_and_numify(train_iter, tokenizer, vocab=vocab).to(device)
    val_data = tokenize_and_numify(val_iter, tokenizer, vocab=vocab).to(device)

    return prepare_data_loader(train_data, val_data, batch_size, vocab, bptt)
