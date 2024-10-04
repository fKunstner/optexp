import subprocess
import sys
import textwrap

import pytest

from optexp import Experiment, Problem
from optexp.datasets import DummyRegression
from optexp.hardwareconfig import StrictManualConfig
from optexp.metrics import Accuracy, CrossEntropy
from optexp.models.linear import Linear

# noinspection PyUnresolvedReferences
from optexp.optim import SGD, DecayEverything  # pylint: disable=unused-import


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
        steps=4,
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
        from optexp.pipes.pipe import TensorDataPipe
        
        experiment={experiment}
        result = run_experiment(experiment)
        
        print("Model parameters")
        for n, p in result.model.named_parameters():
            print(n)
            print(p)
        """
    )


@pytest.mark.skip(reason="Test unreliable")
def test_seed_initialization(tmp_path_factory):

    experiments = [
        make_toy_experiment(num_devices=1),
        make_toy_experiment(num_devices=2),
    ]

    outputs = []
    for i, exp in enumerate(experiments):

        path = tmp_path_factory.mktemp("data") / f"exp{i}.py"
        with open(path, "w") as f:
            f.write(create_python_run_file(exp))

        output = subprocess.check_output(["python", str(path)])

        outpath = tmp_path_factory.mktemp("data") / f"exp{i}.out"
        with open(outpath, "w") as f:
            f.write(output.decode(sys.stdout.encoding))

        outputs.append(output)

    assert outputs[0] == outputs[1]
