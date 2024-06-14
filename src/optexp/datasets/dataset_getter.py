from typing import Optional

from optexp import config
from optexp.datasets.loaders import (
    load_imagenet,
    load_mnist,
    load_ptb,
    load_tiny_stories,
    load_wt2,
    load_wt103,
)


def get_dataset(
    dataset_name,
    batch_size,
    split_prop,
    shuffle,
    num_workers,
    normalize,
):
    raise NotImplementedError()


def get_image_dataset(
    dataset_name, batch_size, shuffle, num_workers, normalize, flatten
):
    if dataset_name == "ImageNet":
        return load_imagenet(
            save_path=config.get_dataset_directory(),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            normalize=normalize,
            flatten=flatten,
            device=config.get_device(),
        )
    elif dataset_name == "MNIST":
        return load_mnist(
            save_path=config.get_dataset_directory(),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            normalize=normalize,
            flatten=flatten,
            device=config.get_device(),
        )
    else:
        raise Exception(f"No Image dataset with name {dataset_name} available.")


def get_text_dataset(
    dataset_name,
    batch_size,
    tgt_len,
    merge: Optional[int] = None,
    mixed_batch_size: Optional[bool] = False,
    eval_batch_size: Optional[int] = 0,
):
    if dataset_name == "PTB":
        return load_ptb(
            save_path=config.get_dataset_directory(),
            batch_size=batch_size,
            bptt=tgt_len,
            device=config.get_device(),
            merge=merge,
        )
    elif dataset_name == "WikiText2":
        return load_wt2(
            save_path=config.get_dataset_directory(),
            batch_size=batch_size,
            bptt=tgt_len,
            device=config.get_device(),
            merge=merge,
        )
    elif dataset_name == "WikiText103":
        return load_wt103(
            save_path=config.get_dataset_directory(),
            tokenizers_path=config.get_tokenizers_directory(),
            batch_size=batch_size,
            bptt=tgt_len,
            device=config.get_device(),
            mixed_batch_size=mixed_batch_size,
            eval_batch_size=eval_batch_size,
        )
    elif dataset_name == "TinyStories":
        return load_tiny_stories(
            save_path=config.get_dataset_directory(),
            tokenizers_path=config.get_tokenizers_directory(),
            batch_size=batch_size,
            bptt=tgt_len,
            device=config.get_device(),
        )
    else:
        raise Exception(f"No Text dataset with name {dataset_name}")
