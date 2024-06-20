import datetime
import time
from pathlib import Path
from typing import Callable, Iterable, Optional, TypeVar

import lightning as ptl
import torch

from optexp.loggers import DataLogger

if __name__ == "__main__":

    fabric = ptl.Fabric(
        accelerator="cpu",
        devices=2,
        num_nodes=1,
        strategy="ddp",
    )

    fabric.launch()
    datalogger = None
    if fabric.global_rank == 0:
        datalogger = DataLogger(
            config_dict={
                "run_date": "{date:%Y-%m-%d_%H-%M-%S}".format(
                    date=datetime.datetime.now()
                )
            },
            group="test_group",
            run_id="runid-{date:%Y-%m-%d_%H-%M-%S}".format(
                date=datetime.datetime.now()
            ),
            exp_id="expid",
            save_directory=Path("."),
            use_wandb=True,
            wandb_autosync=False,
        )
    fabric.barrier()

    for epoch in range(20):
        for iter in range(10):
            if datalogger is not None and fabric.global_rank == 0:
                datalogger.log_data(
                    {
                        "epoch": epoch,
                        "step": epoch * 10 + iter,
                        "loss": 1 / (epoch * 10 + iter + 1),
                    }
                )
                datalogger.commit()
            time.sleep(0.1)
            fabric.barrier()

        if epoch == 10:
            if fabric.global_rank == 1:
                raise ValueError()
            time.sleep(0.1)
            fabric.barrier()

    if datalogger is not None and fabric.global_rank == 0:
        datalogger.finish(exit_code=0)
