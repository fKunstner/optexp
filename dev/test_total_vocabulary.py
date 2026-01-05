import matplotlib.pyplot as plt
import torch

from optexp.datasets import WikiText2
from optexp.datasets.text.tokenizers import BPETokenizer


def load_data():
    vocab_sizes = [
        500,
        1000,
        #        2000,
        5000,
        10000,
        #        20000,
        50000,
    ]
    datasets = []
    for vocab_size in vocab_sizes:
        datasets.append(
            WikiText2(
                sequence_length=1,
                tokenizer=BPETokenizer(vocab_size=vocab_size),
                raw=True,
            )
        )

    return {"vocab_sizes": vocab_sizes, "datasets": datasets}


def postprocess(data):
    return data


def settings(plt):
    pass


def make_figure(fig, data):
    vocab_sizes = data["vocab_sizes"]
    datasets = data["datasets"]

    n = len(datasets)
    axes = [fig.add_subplot(n, 1, i + 1) for i in range(n)]

    if False:
        for vocab_size, dataset in zip(vocab_sizes, datasets):
            tokens = dataset.get_tokens("tr")
            print("Requested vocab size: ", vocab_size)
            print("# Unique tokens in dataset: ", len(tokens.unique()))
            max_id = tokens.max()
            print("Maximum token ID in dataset: ", max_id)
            all_tokens = set(range(max_id))
            print(
                "Tokens with no data in dataset: ",
                sorted(all_tokens - set(tokens.unique().tolist())),
            )

    if True:
        for i, (vocab_size, dataset) in enumerate(zip(vocab_sizes, datasets)):
            counts = dataset.class_counts("tr")
            axes[i].plot(
                range(1, 1 + len(counts)),
                sorted(counts, reverse=True),
                label=f"vocab_size={vocab_size}",
            )
            reweighted_counts = (
                torch.tensor(sorted(counts, reverse=True))
                * torch.tensor(range(1, 1 + len(counts)))
                / len(counts)
            )
            axes[i].plot(
                range(1, 1 + len(counts)),
                reweighted_counts,
                label=f"vocab_size={vocab_size}",
            )

            axes[i].set_yscale("log")
            axes[i].set_xscale("log")

        axes[-1].set_xlabel("Token ID")
        for i in range(n):
            axes[i].set_xlim([1, max(vocab_sizes)])
        for i, vocab_size in enumerate(vocab_sizes):
            axes[i].set_ylabel(f"vocab\n{vocab_size}")
        for i in range(n - 1):
            axes[i].set_xticklabels([])
            axes[i].set_xticks([], minor=False)
            axes[i].set_xticks([], minor=True)

    fig.tight_layout()


if __name__ == "__main__":
    make_figure(plt.figure(), load_data())
    plt.show()
