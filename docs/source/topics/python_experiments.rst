Benefits of defining experiments in Python
==========================================

Defining the experiments in pure Python makes it possible
reuse experiment configuration code and avoid bugs.

Ever wrote an experiment configuration in a ``.json`` file as

.. code-block:: json

   {
      "dataset": "mnist",
      "optim": {"name": "sgd", "lr": 0.01},
      "batch-size": 1000,
   }

only to realize that ``batch-size`` should have been ``batchsize``
and you just ran hours of compute on the default batch size of ``batchsize=128`` instead?
As the experiment definitions are written in Python, you can check for common mistakes without even running them.
Your IDE will tell you about errors and you can even run a static type like `mypy <https://mypy.readthedocs.io/>`_.
