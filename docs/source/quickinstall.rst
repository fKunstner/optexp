Quick Install
===================

This tutorial will get you started with a minimal setup of Optexp.

Down the line, see the
:doc:`installation and configuration guide </topics/install>`
to customize the installation for your system and integrate with Slurm clusters.

Prerequisites
-------------

1. Python 3.10

   Make sure you have an installation of `Python 3.10 or later <https://www.python.org/downloads/>`_.
   Using `venv <https://docs.python.org/3/library/venv.html>`_ or `conda <https://docs.conda.io/en/latest/>`_
   is recommended.

2. Pytorch 2.3.0

   The Pytorch installation is os and hardware dependent, follow
   `Pytorch Installation <https://pytorch.org/get-started/locally/>`_.

Installation
--------------

Installing from source is recommended as the library is not yet stable.

1. Clone the repository
2. Install the dependencies
3. Install the library in editable mode

.. code-block:: bash

   git clone https://github.com/fKunstner/optexp optexp
   cd optexp
   pip install -r requirements/main.txt
   pip install -e .

Once installed, use ``git pull`` to update to the latest version.


Configuration
-------------

Optexp needs to know a few things about your system to run experiments.

1. Where to store dataset files and logs
2. Where to upload the results of the experiments
3. If using a Slurm cluster, the configuration for the cluster

Those are configured through environment variables
and best managed with an environment file.
Here is a template you can copy and fill in.



.. tabs::
   .. group-tab:: OSX / Unix

      .. literalinclude:: ../../envs/env.sh.example
         :language: bash
         :linenos:
         :caption: env.sh


   .. group-tab:: Windows

      .. literalinclude:: ../../envs/env.bat.example
         :language: batch
         :linenos:
         :caption: env.bat

Once complete, you can load the environment variables with
``source env.sh`` (Unix/OSX) or ``call env.bat`` (Windows).

The environment variables are only loaded for the current terminal session
and you will need to load the environment variables again with ``source env.sh`` or ``call env.bat``
on your next session.
To avoid the hassle, add ``source /path/to/env.sh``
to your ``.bashrc`` (Unix) or ``.bash_profile`` (OSX).


Optexp configuration
^^^^^^^^^^^^^^^^^^^^

``OPTEXP_WORKSPACE`` is the directory where the logs and datasets will be stored.
It will fill up, so avoid putting it in a directory with limited space
(eg a university home directory with 5mb of space).
For testing purposes, something like ``~/workspace`` should work fine.

Wandb configuration
^^^^^^^^^^^^^^^^^^^

Optexp uses `Weights and Biases (wandb) <https://wandb.ai/site>`_ to log the results of the experiments.
If you do not have a wandb account, create a one at `wandb.ai <https://wandb.ai>`_.

To log to your account, create a project at https://wandb.ai/$your_username/projects and set

- ``OPTEXP_WANDB_ENTITY`` to your username
- ``OPTEXP_WANDB_PROJECT`` to your project.

The logs will appear at https://wandb.ai/$your_username/$your_project


Next steps
----------

The library is installed and configured!
You can already try to run the :doc:`tutorials </tutorials/index>`.

The Slurm configuration is not necessary for local runs.



Slurm configuration
-------------------

To run experiments using a Slurm cluster,
Optexp and its requirements need to be installed on the login node
with the following tweaks:

- The Pytorch installation on your cluster is likely different, check your cluster's documentation.
- ``OPTEXP_WORKSPACE`` needs to be accessible from the login node *and* the compute nodes.
- ``OPTEXP_SLURM_ACCOUNT`` is the account to charge the jobs to
  (``--account`` in `sbatch <https://slurm.schedmd.com/sbatch.html>`_).
- ``OPTEXP_SLURM_NOTIFICATION_EMAIL`` is the email to receive notifications
  (``--mail-user`` in `sbatch <https://slurm.schedmd.com/sbatch.html>`_).
  Leave it undefined to not receive emails.
