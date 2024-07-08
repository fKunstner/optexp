from small_gridsearch import experiments

from optexp.results.wandb_data_logger import load_wandb_results

if __name__ == "__main__":
    exp_data = load_wandb_results(experiments)

    print(exp_data[experiments[0]])
