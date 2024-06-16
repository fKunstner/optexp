from typing import Callable, Iterable, Optional, TypeVar

import lightning as ptl
import torch

if __name__ == "__main__":

    fabric = ptl.Fabric(
        accelerator="cpu",
        devices=2,
        num_nodes=1,
        strategy="ddp",
    )

    fabric.launch()

    try:
        print("Rank", fabric.global_rank, "a")

        def risky():
            if fabric.global_rank == 1:
                raise ValueError()

        synchronized_try_except(risky)

        print("Rank", fabric.global_rank, "b")

    except SynchronizedException as e:
        if fabric.global_rank == 0:
            print("Rank 0 saw error")
            print(e)
        fabric.barrier()
        raise SystemExit

    print("done", fabric.global_rank)
