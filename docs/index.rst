:github_url: https://github.com/deepmind/sonnet/tree/v2/docs

Sonnet Documentation
====================

Sonnet is a library built on top of TensorFlow designed to provide simple,
composable abstractions for machine learning research.

.. code-block:: python

    import sonnet as snt
    import tensorflow as tf

    mlp = snt.nets.MLP([1024, 1024, 10])
    logits = mlp(tf.ones([8, 28 * 28]))

Installation
------------

Install Sonnet by running::

    $ pip install tensorflow
    $ pip install dm-sonnet

.. toctree::
   :caption: Guides
   :maxdepth: 1

   modules

.. toctree::
   :caption: Package Reference
   :maxdepth: 1

   api

Contribute
----------

- Issue tracker: https://github.com/deepmind/sonnet/issues
- Source code: https://github.com/deepmind/sonnet/tree/v2

Support
-------

If you are having issues, please let us know by filing an issue on our
`issue tracker <https://github.com/deepmind/sonnet/issues>`_.

License
-------

Sonnet is licensed under the Apache 2.0 License.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
