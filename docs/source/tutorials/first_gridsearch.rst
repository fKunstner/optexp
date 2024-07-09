Writing your first grid search
==============================

This tutorial assumes you have completed the :doc:`quick install </quickinstall>`.

Say we want to find a good step-size for SGD
for training a small neural network on MNIST.
The following code defines
3 experiments, one for each step-size we will try;


.. literalinclude:: ../../../examples/small_gridsearch.py
   :caption: gridsearch.py

To download the datasets, run the following command:

.. code-block:: bash

    python gridsearch.py prepare

To check the current status of the experiments, run the following command:

.. code-block:: bash

    python gridsearch.py check
    > Out of 3 experiments, 3 still need to run (0.00% complete)

To run the experiments locally and upload the results to wandb, run the following command:

.. code-block:: bash

    python gridsearch.py run --local

Once the experiments are done, the status of the experiments should read 100% complete:

.. code-block:: bash

    python gridsearch.py check
    > Out of 3 experiments, 3 still need to run (100.00% complete)

You should see the results on wandb
(https://app.wandb.ai/your_username/your_project/runs)
and are ready to move on to :doc:`analyzing the results </tutorials/results>`.

