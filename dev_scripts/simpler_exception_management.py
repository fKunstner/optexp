import torch
from lightning import Fabric

if __name__ == "__main__":

    fabric = Fabric(
        accelerator="cpu",
        devices=2,
        num_nodes=1,
        strategy="ddp",
    )

    fabric.launch()

    all_exceptions = [None for _ in range(fabric.world_size)]
    try:
        # raise an exception on rank 1
        if fabric.global_rank == 1:
            raise ValueError()
    except Exception as exception:
        torch.distributed.all_gather_object(
            obj=exception,
            object_list=all_exceptions,
        )
        raise exception
    else:
        torch.distributed.all_gather_object(
            obj=None,
            object_list=all_exceptions,
        )
    finally:
        valid_exception = list(filter(lambda x: x is not None, all_exceptions))
        if len(valid_exception) > 0:
            if fabric.global_rank == 0:
                print("Found exception", valid_exception)
            raise SystemExit()
