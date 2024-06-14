from optexp import config
from optexp.datasets.loaders import (
    download_mnist,
    download_ptb,
    download_tiny_stories,
    download_wt2,
    download_wt103,
)


def download_dataset(dataset_name):
    if dataset_name == "dummy_class" or dataset_name == "MNIST":
        download_mnist(config.get_dataset_directory())
    elif dataset_name == "PTB":
        download_ptb(config.get_dataset_directory())
    elif dataset_name == "WikiText2":
        download_wt2(config.get_dataset_directory())
    elif dataset_name == "WikiText103":
        download_wt103(config.get_dataset_directory())
    elif dataset_name == "ImageNet":
        print("Downloading ImageNet takes days. Find your own.")
    elif dataset_name == "TinyStories":
        download_tiny_stories(config.get_dataset_directory())
    else:
        raise Exception(f"No dataset with name {dataset_name} available.")
