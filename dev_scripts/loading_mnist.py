import time

import torch
import torchvision
from PIL import Image

from optexp.datasets.image import MNIST


def timecall(func):
    start = time.time()
    func()
    end = time.time()
    print(f"Time taken: {end - start:.2f}s")


if __name__ == "__main__":

    def run_with(num_workers, b=1000, preprocess=False, move_to=None):
        dataset = MNIST()
        dataloader = dataset.get_dataloader(
            b, "tr", num_workers=num_workers, preprocess=preprocess, move_to=move_to
        )
        for x, y in dataloader:
            x, y = x.to("cuda"), y.to("cuda")
            pass

    def do_the_dumb_thing():
        dataset = MNIST()._get_dataset("tr")
        X, targets = dataset.data, dataset.targets

        imgs = X / 255.0

        if dataset.transform is not None:
            imgs = torchvision.transforms.Normalize(mean=0.1307, std=0.3081)(imgs)

        if dataset.target_transform is not None:
            targets = dataset.target_transform(targets)

        imgs = imgs.to("cuda")
        targets = targets.to("cuda")
        return imgs, targets

    print("Just dump the tensor")
    timecall(do_the_dumb_thing)

    for num_workers in [0, 4]:
        print(f"num_workers={num_workers} (small batch)")
        timecall(lambda: run_with(num_workers, b=1000, preprocess=True))
    for num_workers in [0, 4]:
        print(f"num_workers={num_workers} (small batch)")
        timecall(lambda: run_with(num_workers, b=1000, preprocess=True, move_to="cuda"))

    for num_workers in [0, 4]:
        print(f"num_workers={num_workers} (small batch)")
        timecall(lambda: run_with(num_workers, b=1000, preprocess=False))
