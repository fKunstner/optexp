import argparse
import json
import subprocess

from tqdm import tqdm

from optexp.config import Config
from optexp.results.wandb_api import WandbAPI


def sync(upload_every: int, upload_offset: int):
    to_sync_dir = Config.get_workspace_directory() / "syncing" / "to_sync"
    synced_dir = Config.get_workspace_directory() / "syncing" / "synced"
    synced_dir.mkdir(parents=True, exist_ok=True)

    files_to_sync = list(to_sync_dir.glob("*.json"))

    files_to_sync_this_run = []
    for i, file in enumerate(files_to_sync):
        if i % upload_every == upload_offset:
            files_to_sync_this_run.append(file)

    not_already_synced = [
        file for file in files_to_sync_this_run if not (synced_dir / file.name).exists()
    ]

    n_files = len(files_to_sync_this_run)
    n_to_sync = len(not_already_synced)
    n_synced = n_files - n_to_sync

    print(f"Found {n_files} files in {to_sync_dir}.")
    print(f"{n_synced}/{n_files} already synced. {n_to_sync} to process.")

    for file in tqdm(not_already_synced, total=len(not_already_synced)):
        with open(file, "r", encoding="utf-8") as f:
            info = json.load(f)

        folder_to_sync = info["folder"]
        subprocess.run(f"wandb sync {folder_to_sync}", shell=True, check=False)

        # filename format: offline-run-YYYYMMDD_HHMMSS-runid
        _offline, _run, _date, run_id = file.name.split("-")
        run_id = run_id.replace(".json", "")

        try:
            WandbAPI.get_handler().run(
                f"{Config.get_wandb_entity()}/{Config.get_wandb_project()}/{run_id}"
            )
        except ValueError as e:
            print("Syncing failed for", file.name, e)
            print(e)
            print("Skipping this file...")
            continue

        (synced_dir / file.name).touch()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--sync",
        action="store_true",
        help="Sync the results with wandb",
        default=False,
    )
    parser.add_argument(
        "--upload-every",
        type=int,
        help="Upload every nth run",
        default=1,
    )
    parser.add_argument(
        "--upload-offset",
        type=int,
        action="store",
        help="Start uploading at the nth run",
        default=0,
    )

    args = parser.parse_args()

    if args.sync:
        sync(args.upload_every, args.upload_offset)
