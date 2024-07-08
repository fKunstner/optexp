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
