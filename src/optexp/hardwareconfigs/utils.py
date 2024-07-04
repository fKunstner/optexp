from optexp.datasets.dataset import TrVa


def batchsize_mismatch_message(trva: TrVa, n, b):
    dataloader = "training dataloader" if trva == "tr" else "validation dataloader"
    return (
        f"Error in the batch size for {dataloader}."
        "Batch size must divide number of training samples."
        f"Got batch size: {b}, number of training samples: {n}"
    )
