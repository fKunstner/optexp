from optexp.datasets.dataset import TrVaTe


def batchsize_mismatch_message(trvate: TrVaTe, n, b):
    if trvate == "tr":
        dataloader = "training dataloader"
    elif trvate == "va":
        dataloader = "validation dataloader"
    else:
        dataloader = "test dataset"
    return (
        f"Error in the batch size for {dataloader}."
        "Batch size must divide number of training samples."
        f"Got batch size: {b}, number of training samples: {n}"
    )
