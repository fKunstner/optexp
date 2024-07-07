import subprocess
import textwrap

from optexp import Experiment, Problem
from optexp.datasets import DummyRegression
from optexp.hardwareconfig import StrictManualConfig
from optexp.metrics import Accuracy, CrossEntropy
from optexp.models.linear import Linear
from optexp.optim import SGD


def make_toy_experiment(num_devices):
    return Experiment(
        optim=SGD(lr=0.01),
        problem=Problem(
            dataset=DummyRegression(),
            model=Linear(),
            lossfunc=CrossEntropy(),
            metrics=[CrossEntropy(), Accuracy()],
            batch_size=10,
        ),
        group="testing",
        eval_every=10,
        seed=0,
        steps=1,
        hardware_config=StrictManualConfig(
            device="cpu",
            num_devices=num_devices,
        ),
    )


def create_python_run_file(experiment):
    return textwrap.dedent(
        f"""
        from optexp.runner.runner import run_experiment
        from optexp import Experiment, Problem
        from optexp.datasets import DummyRegression
        from optexp.hardwareconfig import StrictManualConfig
        from optexp.metrics import Accuracy, CrossEntropy
        from optexp.models.linear import Linear
        from optexp.optim import SGD, DecayEverything
        
        experiment={experiment}
        result = run_experiment(experiment)
        
        print("Model parameters")
        for n, p in result.model.named_parameters():
            print(n)
            print(p)
        """
    )


def test_seed_initialization(tmp_path_factory):

    exp2 = make_toy_experiment(num_devices=2)
    exp1 = make_toy_experiment(num_devices=1)

    path1 = tmp_path_factory.mktemp("data") / "exp1.py"
    path2 = tmp_path_factory.mktemp("data") / "exp2.py"

    with open(path1, "w") as f:
        f.write(create_python_run_file(exp1))

    with open(path2, "w") as f:
        f.write(create_python_run_file(exp2))

    out1 = subprocess.check_output(["python", str(path1)])
    out2 = subprocess.check_output(["python", str(path2)])

    assert out1 == out2
