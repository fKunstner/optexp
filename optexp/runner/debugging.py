import datetime

import torch

TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"
MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT: int = 100000


def trace_handler(prof: torch.profiler.profile):
    timestamp = datetime.datetime.now().strftime(TIME_FORMAT_STR)
    file_prefix = f"localhost_{timestamp}"
    prof.export_chrome_trace(f"{file_prefix}.json.gz")
    prof.export_memory_timeline(f"{file_prefix}.html", device="cuda:0")


def start_record_memory_history() -> None:
    if not torch.cuda.is_available():
        print("CUDA unavailable. Not recording memory history")
        return

    print("Starting snapshot record_memory_history")
    torch.cuda.memory._record_memory_history(  # pylint: disable=protected-access
        max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT
    )


def stop_record_memory_history() -> None:
    if not torch.cuda.is_available():
        print("CUDA unavailable. Not recording memory history")
        return

    print("Stopping snapshot record_memory_history")
    torch.cuda.memory._record_memory_history(  # pylint: disable=protected-access
        enabled=None
    )


def export_memory_snapshot() -> None:
    if not torch.cuda.is_available():
        print("CUDA unavailable. Not exporting memory snapshot")
        return

    timestamp = datetime.datetime.now().strftime(TIME_FORMAT_STR)
    file_prefix = f"localhost_{timestamp}"

    try:
        print(f"Saving snapshot to local file: {file_prefix}.pickle")
        torch.cuda.memory._dump_snapshot(  # pylint: disable=protected-access
            f"{file_prefix}.pickle"
        )

    except Exception as e:
        print(f"Failed to capture memory snapshot {e}")
        raise e
