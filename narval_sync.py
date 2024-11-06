import os
import subprocess
from tqdm import tqdm
from argparse import ArgumentParser


def main(log_folder):
    slurm_logs = [f for f in os.listdir(log_folder) if f.endswith(".out")]
    for log in tqdm(slurm_logs):
        with open(f"{log_folder}/{log}", "r") as f:
            lines = f.readlines()
            sync_line = None
            for line in lines:
                if "Not Running:    wandb sync" in line:
                    sync_line = line.strip()
                    break
                if "wandb sync --id" in line:
                    sync_line = line.strip()
                    break
            if sync_line:
                cmd = sync_line[sync_line.find("wandb"):].split()
                exp_folder = cmd[-1].split("/")[-1]
                if os.path.exists(f"/home/alanmil/optexp/experiments/wandb/{exp_folder}/run-{exp_folder.split('-')[-1]}.wandb.synced"):
                    continue
                else:
                    try:
                        subprocess.run(cmd, check=True)
                    except:
                        print("Uh Oh, something went wrong when  running {cmd}")
            else:
                print(f"Uh Oh, no wandb sync found for {log_folder}/{log}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--log_folder", type=str, required=True)
    log_folder = parser.parse_args().log_folder
    main(log_folder)
