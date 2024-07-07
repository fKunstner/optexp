Overview
==================

OptExp is a library to manage experiments in optimization for ML.
Defining and running experiments can be tedious and error-prone,
and Optexp aims to make those tasks easier and reduce wasted time due to common errors.
The library provides utilities to

- Define experiments, directly in Python code
- Prepare the necessary components, such as downloading the datasets
- Run experiments

   - locally or on a slurm cluster
   - log the results to `wandb <https://wandb.ai>`_
   - with support for multiple GPUs

- Check that all experiments have succeeded and logged their data
- Download the data locally to make plots


Defining experiments
----------------------

Optexp defines experiments directly in Python code.
For example, to run a grid search for the step-size of SGD:

.. literalinclude:: ../../examples/small_gridsearch.py


Running experiments
----------------------

Optexp provides a command line interface to manage experiments.

1. Prepare experiments and make sure the datasets are downloaded

   .. code-block:: bash

       python experiments.py prepare


2. Run experiments either locally or on a slurm cluster.
   The results will be logged to `wandb <https://wandb.ai>`_.

   .. code-block:: bash

       python experiments.py run --local
       python experiments.py run --slurm

3. Check that all experiments have succeeded and logged their data.

   .. code-block:: bash

       python experiments.py check

4. Download the data locally to make plots

   .. code-block:: bash

       python experiments.py download

Load the results
------------------------

All the experiments have run and it's time to make plots?

.. code-block:: python

   from experiments import experiments

   exp_data = [load_data(exp) for exp in experiments]

Filter experiments directly in Python

.. code-block:: python

   filtered_data = [
       load_data(exp) for exp in experiments
       if exp.optimizer.lr == 0.01
   ]




