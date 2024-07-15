Adding a dataset
================

To add your own dataset or integrate datasets from other libraries,
you can create new component that extend the :class:`~optexp.datasets.Dataset` class.

This interface specifies abstract methods that need to be implemented:

.. literalinclude:: ../../../optexp/datasets/dataset.py
   :pyobject: Dataset

As an example, we'll follow the current implementation for :class:`~optexp.datasets.mnist.MNIST`.

Basic information about the dataset
------------------------------------

The following 3 methods just return basic statistics about the dataset:

- ``input_shape``

   .. literalinclude:: ../../../optexp/datasets/dataset.py
      :pyobject: Dataset.input_shape

   .. literalinclude:: ../../../optexp/datasets/mnist.py
      :pyobject: MNIST.input_shape

- ``output_shape``

   .. literalinclude:: ../../../optexp/datasets/dataset.py
      :pyobject: Dataset.output_shape

   .. literalinclude:: ../../../optexp/datasets/mnist.py
      :pyobject: MNIST.output_shape

- ``get_num_samples``

   .. literalinclude:: ../../../optexp/datasets/dataset.py
      :pyobject: Dataset.get_num_samples

   .. literalinclude:: ../../../optexp/datasets/mnist.py
      :pyobject: MNIST.get_num_samples

Making dataloaders
-------------------

The last method, ``get_dataloader``, returns the actual :external:obj:`torch.utils.data.DataLoader`
that will be used to iterate over the dataset.

   .. literalinclude:: ../../../optexp/datasets/dataset.py
      :pyobject: Dataset.get_dataloader

   .. literalinclude:: ../../../optexp/datasets/mnist.py
      :pyobject: MNIST.get_dataloader

   .. literalinclude:: ../../../optexp/datasets/mnist.py
      :pyobject: MNIST._get_dataset

   .. literalinclude:: ../../../optexp/datasets/utils.py
      :pyobject: make_dataloader



Adding further integrations
-----------------------------

Additional features can be implemented by inheriting from the following classes.
For example, to define a dataset that can be downloaded from the internet
when using the ``prepare`` CLI option, extend both :class:`~optexp.datasets.dataset.Dataset`
and :class:`~optexp.datasets.dataset.Downloadable`
and implement their respective abstract methods;

.. code:: python

   class MyDataset(Dataset, Downloadable):
       ...

The :class:`~optexp.datasets.dataset.Downloadable` extension is useful,
the other are very much optional.

Downloadble datasets
++++++++++++++++++++++

.. literalinclude:: ../../../optexp/datasets/dataset.py
   :pyobject: Downloadable

.. literalinclude:: ../../../optexp/datasets/mnist.py
   :pyobject: MNIST.download

.. literalinclude:: ../../../optexp/datasets/mnist.py
   :pyobject: MNIST.is_downloaded

Datasets with class counts
++++++++++++++++++++++++++++++

.. literalinclude:: ../../../optexp/datasets/dataset.py
   :pyobject: HasClassCounts

.. literalinclude:: ../../../optexp/datasets/mnist.py
   :pyobject: MNIST.class_counts

Datasets that can be put in RAM
++++++++++++++++++++++++++++++++++

.. literalinclude:: ../../../optexp/datasets/dataset.py
   :pyobject: InMemory

.. literalinclude:: ../../../optexp/datasets/mnist.py
   :pyobject: MNIST._get_tensor_dataset

.. literalinclude:: ../../../optexp/datasets/mnist.py
   :pyobject: MNIST.get_in_memory_dataloader

Datasets that need to be moved to local storage
++++++++++++++++++++++++++++++++++++++++++++++++

.. literalinclude:: ../../../optexp/datasets/dataset.py
   :pyobject: MovableToLocal

Not implemented by MNIST as the dataset is small.

Full MNIST Implementation
-------------------------

.. literalinclude:: ../../../optexp/datasets/mnist.py
   :pyobject: MNIST

