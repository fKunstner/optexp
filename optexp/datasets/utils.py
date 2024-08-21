from typing import Any, List

import torch


def make_dataloader(dataset, b, num_workers):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=b,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
    )


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, inputs: List[Any], targets: List[Any]):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx: int):
        return self.inputs[idx], self.targets[idx]
