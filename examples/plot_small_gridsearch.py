from small_gridsearch import experiments

if __name__ == "__main__":
    exp_data = [exp.load_data() for exp in experiments]
