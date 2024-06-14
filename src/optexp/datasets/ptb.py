from dataclasses import dataclass, field

from optexp.datasets import Dataset


@dataclass(frozen=True)
class PTB(Dataset):
    name: str = field(default="MNIST", init=False)
    target_len: int = 35

    def load(self):
        pass

    def download(self):
        pass
