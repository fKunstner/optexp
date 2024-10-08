"""Module to integrate with Slurm."""

import math
import textwrap
from pathlib import Path
from typing import List

from optexp.config import Config
from optexp.runner.slurm.slurm_config import SlurmConfig


def make_sbatch_header(slurm_config: SlurmConfig, n_jobs: int) -> str:
    """Generates the header of a sbatch file for Slurm.

    Args:
        slurm_config: Slurm configuration to use
        n_jobs: Number of jobs to run in the batch
    """
    if Config.slurm_email is None:
        email_lines = []
    else:
        email_lines = [
            f"#SBATCH --mail-user={Config.slurm_email}",
            "#SBATCH --mail-type=ALL",
        ]

    header_lines = [
        "#!/bin/sh",
        f"#SBATCH --account={Config.get_slurm_account()}",
        f"#SBATCH --mem={slurm_config.mem_str}",
        f"#SBATCH --time={slurm_config.time_str}",
        f"#SBATCH --cpus-per-task={slurm_config.n_cpus}",
        *email_lines,
        f"#SBATCH --array=0-{n_jobs - 1}",
        slurm_config.gpu_str,
        "",
    ]
    return "\n".join(header_lines)


def make_jobarray_content(
    run_exp_by_idx_command: str, should_run: List[bool], jobs_per_node: int
):
    """Creates the content of a jobarray sbatch file for Slurm.

    Args:
        run_exp_by_idx_command: Command to call to run the i-th experiments
        should_run: Whether the matching experiments should run
        jobs_per_node: How many experiments to run (in sequence) on a slurm node
    """

    bash_script_idx_to_exp_script_idx = []
    for i, _should_run in enumerate(should_run):
        if _should_run:
            bash_script_idx_to_exp_script_idx.append(i)

    commands_for_each_experiment = []

    # for bash_script_idx, exp_script_idx in enumerate(bash_script_idx_to_exp_script_idx):
    n_nodes_required = math.ceil(len(bash_script_idx_to_exp_script_idx) / jobs_per_node)
    for bash_script_idx in range(n_nodes_required):
        for _ in range(jobs_per_node):
            if len(bash_script_idx_to_exp_script_idx) > 0:
                commands_for_each_experiment.append(
                    textwrap.dedent(
                        f"""
                        if [ $SLURM_ARRAY_TASK_ID -eq {bash_script_idx} ]
                        then
                            {run_exp_by_idx_command} {bash_script_idx_to_exp_script_idx.pop(0)}
                        fi
                        """
                    )
                )

    return "".join(commands_for_each_experiment)


def make_jobarray_file_contents(
    experiment_file: Path,
    should_run: List[bool],
    slurm_config: SlurmConfig,
):
    """Creates a jobarray sbatch file for Slurm."""
    n_jobs = math.ceil(sum(should_run) / slurm_config.jobs_per_node)
    header = make_sbatch_header(slurm_config=slurm_config, n_jobs=n_jobs)

    body = make_jobarray_content(
        run_exp_by_idx_command=f"python {experiment_file} run --single",
        should_run=should_run,
        jobs_per_node=slurm_config.jobs_per_node,
    )

    footer = textwrap.dedent(
        """
        exit
        """
    )

    return header + body + footer
