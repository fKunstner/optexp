from dataclasses import dataclass, field

from torchtext.datasets import PennTreebank as TorchPTB
from torchtext.vocab import Vocab, build_vocab_from_iterator

from optexp import Dataset


@dataclass(frozen=True)
class PTB(Dataset):
    name: str = field(default="MNIST", init=False)
    target_len: int = 35

    def load(self):
        pass

    def download(self):
        pass
