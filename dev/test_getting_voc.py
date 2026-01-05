from optexp.datasets import WikiText2
from optexp.datasets.text.tokenizers import BPETokenizer

if __name__ == "__main__":
    dataset = WikiText2(sequence_length=3, tokenizer=BPETokenizer(vocab_size=50000))
    dataset.get_dataloader(10, split="tr", num_workers=0)
