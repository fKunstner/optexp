from pathlib import Path

import pytest

from optexp.datasets import WikiText103
from optexp.datasets.text.tokenizers import MERGE_FILE, VOCAB_FILE, BPETokenizer
from optexp.datasets.text.wikitext import WikiText2


def download_and_run(dataset):
    dataset.download()

    files = dataset._get_files()
    for file in files.all_txt_files():
        assert Path.exists(file)

    dataset.tokenizer.build_tokenizer(
        files.base_path(),
        files.txt_file("tr"),
    )

    assert isinstance(dataset.tokenizer, BPETokenizer)
    assert (dataset.tokenizer._tokenizer_path(files.base_path()) / MERGE_FILE).exists()
    assert (dataset.tokenizer._tokenizer_path(files.base_path()) / VOCAB_FILE).exists()

    dataloader = dataset.get_dataloader(b=512, split="tr", num_workers=4)
    batch, targets = next(iter(dataloader))
    print(batch, targets)


@pytest.mark.long
def test_download_and_tokenize_wikitext2():
    raw = True
    vocab_size = 10000
    dataset = WikiText2(
        raw=raw, sequence_length=1024, tokenizer=BPETokenizer(vocab_size=vocab_size)
    )
    download_and_run(dataset)


@pytest.mark.long
def test_download_and_tokenize_wikitext103():
    raw = True
    vocab_size = 50257
    dataset = WikiText103(
        raw=raw, sequence_length=1024, tokenizer=BPETokenizer(vocab_size=vocab_size)
    )
    download_and_run(dataset)


if __name__ == "__main__":
    test_download_and_tokenize_wikitext2()
    test_download_and_tokenize_wikitext103()
