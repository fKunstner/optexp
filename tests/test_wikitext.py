from pathlib import Path

import torch.nn

from optexp.config import Config
from optexp.datasets import WikiText103
from optexp.datasets.text.tokenizers import BPETokenizer
from optexp.datasets.text.wikitext103 import WTFiles
from optexp.models import Linear


def test_download_and_tokenize_wikitext():
    raw = True
    vocab_size = 50257
    dataset = WikiText103(raw=raw, sequence_length=1024, vocab_size=vocab_size)
    dataset.download()

    wt103_files = WTFiles(raw)
    for file in wt103_files.all_txt_files():
        assert Path.exists(file)

    dataset.tokenizer.build_tokenizer(
        Config.get_dataset_directory() / "WikiText103" / f"wikitext103-raw_v=50257",
        Config.get_dataset_directory() / "WikiText103" / "wiki-raw.train.tokens",
        vocab_size=vocab_size,
    )

    assert isinstance(dataset.tokenizer, BPETokenizer)
    assert dataset.tokenizer._merge_file(vocab_size).exists()
    assert dataset.tokenizer._vocab_file(vocab_size).exists()

    tokens = dataset.get_tokens(tr_va="tr")
    model = Linear().load_model(
        dataset.input_shape(batch_size=512), dataset.output_shape(batch_size=512)
    )
    dataloader = dataset.get_dataloader(b=512, tr_va="tr", num_workers=4)
    batch, targets = next(iter(dataloader))
    print(batch, targets)


if __name__ == "__main__":
    test_download_and_tokenize_wikitext()
