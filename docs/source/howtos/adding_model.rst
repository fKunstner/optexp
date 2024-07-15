Adding a model
==============

To add your own model or integrate models from other libraries,
you can create new component that extend the :class:`~optexp.models.Model` class.

This interface specifies abstract methods that need to be implemented:

.. literalinclude:: ../../../optexp/models/model.py
   :pyobject: Model

As an example, we'll follow the current implementation for :class:`~optexp.models.vision.LeNet5`:

.. literalinclude:: ../../../optexp/models/vision.py
   :pyobject: LeNet5



