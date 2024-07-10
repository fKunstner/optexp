Installation and configuration
==============================




Wandb configuration
-------------------

- ``OPTEXP_WANDB_ENABLED`` (``true`` (recommended), ``false``)
   Controls whether to log to wandb. Should be ``true`` when running experiments,
   but it might be useful to set it to ``false`` when running tests.

   Overrided by the ``--no-wandb`` command line flag.

- ``OPTEXP_WANDB_MODE`` (``offline`` (recommended), ``online``)
   Controls whether to log to wandb in offline or online mode.
   In online mode, the logs are uploaded to the wandb server as they are generated.
   In offline mode, the logs are saved locally and need to be synced.

   When running experiments in parallel on multiple machines,
   it is better to use ``offline`` to avoid hitting the wandb api limits.
   The logs will be uploaded once the experiments are finished.sousa
   When running experiments sequentially on a single machine, ``online`` can be convenient.

   Overrided by the ``--wandb-offline`` and ``--wandb-online`` command line flags.

- ``OPTEXP_WANDB_AUTOSYNC`` (``true`` (recommended), ``false``)
   Controls whether to call ``wandb sync`` automatically after each run.
   Has no effect when ``OPTEXP_WANDB_MODE`` is set to ``online``.

   If running on a worker node with no internet access,
   set it to ``false`` and call ``wandb sync`` manually on a login node once the runs are finished.


   When ``true``, ``wandb sync`` is called automatically after each experiment.
   When ``false``, the logs are saved locally and need to be uploaded manually
   by calling ``wandb sync $path``. The logs indicate the path where they are saved.

   Overrided by the ``--wandb-autosync`` command line flags.



Logging to a team account
^^^^^^^^^^^^^^^^^^^^^^^^^

To log to a shared team project, if you have one,
set the variables
``OPTEXP_WANDB_ENTITY`` to the name of the team
and ``OPTEXP_WANDB_PROJECT`` to the name of the project.
The logs will appear at https://wandb.ai/$team_name/$project_name



Workspace directory
-------------------




