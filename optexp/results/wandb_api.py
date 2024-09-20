from typing import Dict, List

import wandb

from optexp.config import Config, get_logger
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
            WandbAPI.api_handler = wandb.Api(timeout=Config.wandb_timeout)
        return WandbAPI.api_handler

    @staticmethod
    def get_path():
        return f"{Config.get_wandb_entity()}/{Config.get_wandb_project()}"


def _create_wandb_filter(exps):
    """Returns a dictionary of {wandb_key:value} to filter runs on wandb."""
    similar_attributes = {}
    problem0_class = exps[0].problem.__class__
    if all(exp.problem.__class__ == problem0_class for exp in exps):
        similar_attributes["config.exp_config.problem.__class__"] = (
            problem0_class.__name__
        )

    dataset0_class = exps[0].problem.dataset.__class__
    if all(exp.problem.dataset.__class__ == dataset0_class for exp in exps):
        similar_attributes["config.exp_config.problem.dataset.__class__"] = (
            dataset0_class.__name__
        )

    model0_class = exps[0].problem.model.__class__
    if all(exp.problem.model.__class__ == model0_class for exp in exps):
        similar_attributes["config.exp_config.problem.model.__class__"] = (
            model0_class.__name__
        )

    lossfunc0_class = exps[0].problem.lossfunc.__class__
    if all(exp.problem.lossfunc == lossfunc0_class for exp in exps):
        similar_attributes["config.exp_config.problem.lossfunc.__class__"] = (
            lossfunc0_class.__name__
        )

    return similar_attributes


def get_wandb_runs_by_hash(
    experiments: List[Experiment],
    per_page: int = 1000,
) -> Dict[str, List[wandb.apis.public.Run]]:
    """Get all the runs on wandb for a list of experiments."""
    # https://docs.wandb.ai/guides/track/public-api-guide#querying-multiple-runs

    filters = {
        "$and": [{"state": {"$in": ["finished"]}}]
        + [{"tags": {"$in": ["finished"]}}]
        + [{"tags": {"$nin": ["failed_sync"]}}]
        + [{k: v} for k, v in _create_wandb_filter(experiments).items()]
    }

    runs = WandbAPI.get_handler().runs(
        WandbAPI.get_path(), filters=filters, per_page=per_page
    )

    runs_by_hash: Dict[str, List[wandb.apis.public.Run]] = {}
    has_warned = False
    for run in runs:
        if not has_warned and "short_equiv_hash" not in run.config:
            get_logger().warning(
                "Some runs do not have a short_equiv_hash attribute, "
                "and were likely uploaded to wandb without optexp. "
                "If you expect the wandb project to only contain runs generated by optexp, "
                "this might indicate a problem."
            )
            has_warned = True

        if run.config["short_equiv_hash"] not in runs_by_hash:
            runs_by_hash[run.config["short_equiv_hash"]] = []
        runs_by_hash[run.config["short_equiv_hash"]].append(run)

    for exp in experiments:
        if exp.short_equivalent_hash() not in runs_by_hash:
            runs_by_hash[exp.short_equivalent_hash()] = []

    return runs_by_hash


def get_wandb_runs(
    exps: List[Experiment],
) -> dict[Experiment, List[wandb.apis.public.Run]]:
    """Get the runs of all successful runs on wandb for a list of experiments."""
    runs_by_hash = get_wandb_runs_by_hash(exps)
    filtered_runs_by_exp = {
        exp: runs_by_hash[exp.short_equivalent_hash()] for exp in exps
    }
    return filtered_runs_by_exp


def get_wandb_runs_for_group(group: str) -> List[wandb.apis.public.Run]:
    """Get the runs of all successful runs on wandb for a group."""
    # https://docs.wandb.ai/guides/track/public-api-guide#querying-multiple-runs
    runs = WandbAPI.get_handler().runs(
        WandbAPI.get_path(), filters={"group": group}, per_page=1000
    )

    if any("exp_id" not in run.config for run in runs):
        get_logger().warning("Some runs do not have an exp_id attribute.")

    return [run for run in runs if run.state == "finished" and "finished" in run.tags]


def get_successful_ids_and_runs(group: str):
    """Get the experiments ids of all successful runs on wandb for a group."""
    # https://docs.wandb.ai/guides/track/public-api-guide#querying-multiple-runs
    runs = get_wandb_runs_for_group(group)
    successful_runs = []
    successful_exp_ids = []
    for run in runs:
        if (
            run.config["exp_id"] not in successful_exp_ids
            and run.state == "finished"
            and "finished" in run.tags
        ):
            successful_runs.append(run)
            successful_exp_ids.append(run.config["exp_id"])

    return successful_exp_ids, successful_runs
