Metrics
==================

Metrics to evaluate performance and to use as loss function.

Not all metrics are suitable as loss functions.
Some are not differentiable or do not return a scalar.

Metrics are computed on the training and validation sets
every ``eval_freq`` steps, set in :class:`~optexp.experiments.Experiment`.

.. currentmodule:: optexp.metrics
.. autoclass:: Metric


Output-Target Metrics
---------------------

Metrics that are computed on the output of the model and the target.

.. autoclass:: LossLikeMetric
.. autoclass:: Accuracy
.. autoclass:: CrossEntropy



.. autoclass:: AccuracyPerClass
.. autoclass:: CrossEntropyPerClass

