from typing import Callable, List

import wandb

import optexp.config
from optexp.config import get_logger, get_wandb_timeout
from optexp.experiment import Experiment


class WandbAPI:
    """Static class to provide a singleton handler to the wandb api.

    When in need to call the Wandb API, use WandbAPI.get_handler() instead of creating a
    new instance of wandb.Api().
    """

    api_handler = None

    @staticmethod
    def get_handler():
        if WandbAPI.api_handler is None:
            WandbAPI.api_handler = wandb.Api(timeout=get_wandb_timeout())
        return WandbAPI.api_handler

    @staticmethod
    def get_path():
        return f"{optexp.config.get_wandb_entity()}/{optexp.config.get_wandb_project()}"


def get_wandb_runs_by_hash(
    experiments: List[Experiment],
) -> List[wandb.apis.public.Run]:
    """Get all the runs on wandb for a list of experiments."""
    # https://docs.wandb.ai/guides/track/public-api-guide#querying-multiple-runs
    return WandbAPI.get_handler().runs(
        WandbAPI.get_path(),
        filters={
            "$and": {
                "$in": {"state": ["finished"]},
                "$or": [
                    {"config.short_equiv_hash": exp.short_equivalent_hash()}
                    for exp in experiments
                ],
            }
        },
        per_page=1000,
    )


def get_wandb_runs(
    exps: List[Experiment],
) -> dict[Experiment, List[wandb.apis.public.Run]]:
    """Get the runs of all successful runs on wandb for a list of experiments."""

    possible_runs = get_wandb_runs_by_hash(exps)

    def run_matches(exp: Experiment) -> Callable[[wandb.apis.public.Run], bool]:
        def run_match(runs: wandb.apis.public.Run) -> bool:
            if "short_equiv_hash" not in runs.config:
                get_logger().warning(
                    "Some runs do not have a short_equiv_hash attribute."
                )
                return False
            return runs.config["short_equiv_hash"] == exp.short_equivalent_hash()

        return run_match

    return {exp: list(filter(run_matches(exp), possible_runs)) for exp in exps}


def get_wandb_runs_for_group(group: str) -> List[wandb.apis.public.Run]:
    """Get the runs of all successful runs on wandb for a group."""
    # https://docs.wandb.ai/guides/track/public-api-guide#querying-multiple-runs
    runs = WandbAPI.get_handler().runs(
        WandbAPI.get_path(), filters={"group": group}, per_page=1000
    )

    if any("exp_id" not in run.config for run in runs):
        get_logger().warning("Some runs do not have an exp_id attribute.")

    return [run for run in runs if run.state == "finished"]


def get_successful_ids_and_runs(group: str):
    """Get the experiments ids of all successful runs on wandb for a group."""
    # https://docs.wandb.ai/guides/track/public-api-guide#querying-multiple-runs
    runs = get_wandb_runs_for_group(group)
    successful_runs = []
    successful_exp_ids = []
    for run in runs:
        if run.config["exp_id"] not in successful_exp_ids and run.state == "finished":
            successful_runs.append(run)
            successful_exp_ids.append(run.config["exp_id"])

    return successful_exp_ids, successful_runs