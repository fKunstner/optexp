from typing import Callable, Iterable, Optional, TypeVar

import lightning as ptl
import torch


class SynchronizedError(Exception):
    pass


T = TypeVar("T")


def synchronized_try_except(fabric: ptl.Fabric, func: Callable[[], T]) -> Optional[T]:
    """Raise an exception on all ranks if any rank raises an exception.

    The function `func` must not use multiprocess communication (reduce, gather, ...).
    If it does, the processes might deadlock because the following can happen:
    - Rank 0 raises an exception and tries to broadcast it to all ranks.
    - Rank 1 did not see the exception and tries to gather/reduce some tensor with other ranks.

    This function re-raises the exception on all ranks, so that all ranks can handle it
    """
    all_exceptions: Iterable[Optional[Exception]] = [
        None for _ in range(fabric.world_size)
    ]
    all_traces: Iterable[Optional[str]] = [None for _ in range(fabric.world_size)]
    output: Optional[T] = None

    if fabric.world_size == 1:
        try:
            output = func()
        except Exception as e:
            raise SynchronizedError from e

    else:
        try:
            output = func()
        except Exception as exception:
            torch.distributed.all_gather_object(
                obj=exception,
                object_list=all_exceptions,
            )
            torch.distributed.all_gather_object(
                obj=get_trace(exception),
                object_list=all_traces,
            )
            raise exception
        else:
            torch.distributed.all_gather_object(
                obj=None,
                object_list=all_exceptions,
            )
            torch.distributed.all_gather_object(
                obj=None,
                object_list=all_traces,
            )
        finally:
            fabric.barrier()
            if fabric.global_rank == 0:
                for rank, (exc, trace) in enumerate(zip(all_exceptions, all_traces)):
                    if exc is not None:
                        raise SynchronizedError(
                            f"Exception occurred on rank {rank}.\n"
                            f"Traceback for {exc.__class__.__name__}:\n"
                            f"{trace}"
                        ) from exc


def get_trace(ex: BaseException):
    import traceback

    return "".join(traceback.TracebackException.from_exception(ex).format())


if __name__ == "__main__":

    fabric = ptl.Fabric(
        accelerator="cpu",
        devices=2,
        num_nodes=1,
        strategy="ddp",
    )

    fabric.launch()

    def risky():
        if fabric.global_rank == 0:
            raise ValueError()

    synchronized_try_except(fabric, risky)

    print("Rank", fabric.global_rank, "b")
