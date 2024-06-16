"""Slurm runner for OptExp."""

from optexp.slurm.sbatch_writers import make_jobarray_content, make_sbatch_header

__all__ = ["make_sbatch_header", "make_jobarray_content", "SlurmConfig"]
