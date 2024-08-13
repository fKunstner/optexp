from tqdm import tqdm
import torch
from optexp.config import Config
import sentencepiece as sp


def main():
    vocab_size = 50257
    data_path = Config.get_dataset_directory() / "WikiText103" / "wiki-raw.train.tokens"
    sp.SentencePieceTrainer.Train(
        input=data_path,
        model_prefix="test_tokenizer",
        vocab_size=vocab_size,
        model_type="bpe",
    )
    tokenizer = sp.SentencePieceProcessor()
    tokenizer.Load("test_tokenizer.model")

    text_lines = open(data_path, "r").readlines()
    tokenized_lines = []
    for line in tqdm(text_lines):
        tokenized_lines.append(
            torch.tensor(tokenizer.encode_as_ids(line), dtype=torch.long)
        )
    tokens = torch.cat(tokenized_lines)
    torch.save(tokens, f"tokens_v={vocab_size}.pt")

    import pdb

    pdb.set_trace()


if __name__ == "__main__":
    main()
