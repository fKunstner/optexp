import traceback
from typing import Callable, Iterable, List, Optional, Tuple, TypeVar

import lightning as ptl
import torch

from optexp.problems import DivergingException

T = TypeVar("T")


class SynchronizedDivergence(Exception):
    pass


class SynchronizedError(Exception):
    pass


def sync_try_except(fabric: ptl.Fabric, func: Callable[[], T]) -> T:
    """Raise an exception on all ranks if any rank raises an exception.

    The function `func` must not use multiprocess communication (reduce, gather, ...).
    If it does, the processes might deadlock because the following can happen:
    - Rank 0 raises an exception and tries to broadcast it to all ranks.
    - Rank 1 did not see the exception and tries to gather/reduce some tensor with other ranks.

    This function re-raises the exception on all ranks, so that all ranks can handle it
    """
    w = fabric.world_size
    all_exceptions: Iterable[Optional[Tuple[Exception, str]]] = [None for _ in range(w)]
    output: Optional[T] = None

    if fabric.world_size == 1:
        try:
            output = func()
        except DivergingException as e:
            raise SynchronizedDivergence from e
        except Exception as e:
            raise SynchronizedError from e
        return output

    try:
        output = func()
    except Exception as exception:
        torch.distributed.all_gather_object(
            obj=(exception, get_trace(exception)),
            object_list=all_exceptions,
        )
        raise exception
    else:
        torch.distributed.all_gather_object(
            obj=None,
            object_list=all_exceptions,
        )
    finally:
        valid_exceptions: List[Tuple[int, Exception, str]] = []
        for i, exp_and_trace in enumerate(all_exceptions):
            if exp_and_trace is not None:
                valid_exceptions.append((i, exp_and_trace[0], exp_and_trace[1]))

        if len(valid_exceptions) > 0:
            if fabric.global_rank >= 0:
                rank, exc, trace = valid_exceptions[0]
                exc_class = exc.__class__.__name__
                if isinstance(exc, DivergingException):
                    raise SynchronizedDivergence(
                        f"Detected {exc_class} on rank {rank}.\n\n"
                        f"Traceback for {exc_class}:\n{trace}"
                    ) from exc
                raise SynchronizedError(
                    f"Detected {exc_class} on rank {rank}.\n\n"
                    f"Traceback for {exc_class}:\n{trace}"
                ) from exc
            raise SystemExit()

    return output


def get_trace(ex: BaseException):
    return "".join(traceback.TracebackException.from_exception(ex).format())
