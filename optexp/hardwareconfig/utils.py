from optexp.datasets.dataset import Split


def batchsize_mismatch_message(split: Split, n, b):
    if split == "tr":
        dataloader = "training dataloader"
    elif split == "va":
        dataloader = "validation dataloader"
    else:
        dataloader = "test dataset"
    return (
        f"Error in the batch size for {dataloader}. "
        "Batch size must divide number of training samples. "
        f"Got batch size: {b}, number of training samples: {n}"
    )
