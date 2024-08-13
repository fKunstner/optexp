from pathlib import Path

import torch.nn

from optexp.config import Config
from optexp.datasets import WikiText103
from optexp.models import Linear


def test_download_and_tokenize_wikitext():
    dataset = WikiText103(raw=True, sequence_length=1024, vocab_size=50257)
    dataset.download()
    assert Path.exists(
        Config.get_dataset_directory() / "WikiText103" / "wiki-raw.train.tokens"
    )
    assert Path.exists(
        Config.get_dataset_directory() / "WikiText103" / "wiki-raw.valid.tokens"
    )
    assert Path.exists(
        Config.get_dataset_directory() / "WikiText103" / "wiki-raw.test.tokens"
    )
    # dataset.tokenizer.build_tokenizer(
    #     Config.get_dataset_directory() / "WikiText103" / f"wikitext103-raw_v=50257",
    #     Config.get_dataset_directory() / "WikiText103" / "wiki-raw.train.tokens",
    #     vocab_size=50257,
    # )
    assert Path.exists(
        Config.get_dataset_directory() / "WikiText103" / "wikitext103-raw_v=50257.model"
    )
    assert Path.exists(
        Config.get_dataset_directory() / "WikiText103" / "wikitext103-raw_v=50257.vocab"
    )
    tokens = dataset.get_tokens(tr_va="tr", vocab_size=50257)

    model = Linear().load_model(
        dataset.input_shape(batch_size=512), dataset.output_shape(batch_size=512)
    )
    dataloader = dataset.get_dataloader(b=512, tr_va="tr", num_workers=4)
    batch, targets = next(iter(dataloader))
    out = model(batch)
    print(out)


if __name__ == "__main__":
    test_download_and_tokenize_wikitext()
