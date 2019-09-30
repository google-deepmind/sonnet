.. TODO(slebedev): add a title, e.g. "API docs"?

Base
----

.. currentmodule:: sonnet

Module
~~~~~~

.. autoclass:: Module
   :members:

once
~~~~

.. autofunction:: once

no_name_scope
~~~~~~~~~~~~~

.. autofunction:: no_name_scope

Deferred
~~~~~~~~

.. autoclass:: Deferred
   :members:
   :exclude-members: __getattr__, __setattr__

Linear modules
--------------

Linear
~~~~~~

.. autoclass:: Linear
   :members:

Bias
~~~~

.. autoclass:: Bias
   :members:

Convolutional modules
---------------------

.. currentmodule:: sonnet

Conv1D
~~~~~~

.. autoclass:: Conv1D
   :members:

Conv2D
~~~~~~

.. autoclass:: Conv2D
   :members:

Conv3D
~~~~~~

.. autoclass:: Conv3D
   :members:

Conv1DTranspose
~~~~~~~~~~~~~~~

.. autoclass:: Conv1DTranspose
   :members:

Conv2DTranspose
~~~~~~~~~~~~~~~

.. autoclass:: Conv2DTranspose
   :members:

Conv3DTranspose
~~~~~~~~~~~~~~~

.. autoclass:: Conv3DTranspose
   :members:

Normalization modules
---------------------

.. currentmodule:: sonnet

LayerNorm
~~~~~~~~~

.. autoclass:: LayerNorm
   :members:

InstanceNorm
~~~~~~~~~~~~

.. autoclass:: InstanceNorm
   :members:

BaseBatchNorm
~~~~~~~~~~~~~

.. autoclass:: BaseBatchNorm
   :members:

BatchNorm
~~~~~~~~~

.. autoclass:: BatchNorm
   :members:

CrossReplicaBatchNorm
~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: sonnet.distribute
.. autoclass:: CrossReplicaBatchNorm
   :members:

GroupNorm
~~~~~~~~~

.. currentmodule:: sonnet
.. autoclass:: GroupNorm
   :members:

Recurrent modules
-----------------

.. currentmodule:: sonnet

RNNCore
~~~~~~~

.. autoclass:: RNNCore
   :members:

UnrolledRNN
~~~~~~~~~~~

.. autoclass:: UnrolledRNN
   :members:

TrainableState
~~~~~~~~~~~~~~

.. autoclass:: TrainableState
   :members:

dynamic_unroll
~~~~~~~~~~~~~~

.. autofunction:: dynamic_unroll

static_unroll
~~~~~~~~~~~~~

.. autofunction:: static_unroll

VanillaRNN
~~~~~~~~~~

.. autoclass:: VanillaRNN
   :members:
   :special-members:

DeepRNN
~~~~~~~

.. autoclass:: DeepRNN
   :members:

.. autofunction:: deep_rnn_with_skip_connections

.. autofunction:: deep_rnn_with_residual_connections

LSTM
~~~~

.. autoclass:: LSTM
   :members:

.. autoclass:: LSTMState

lstm_with_recurrent_dropout
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: lstm_with_recurrent_dropout

UnrolledLSTM
~~~~~~~~~~~~

.. autoclass:: UnrolledLSTM
   :members:

Conv1DLSTM
~~~~~~~~~~

.. autoclass:: Conv1DLSTM
   :members:

Conv2DLSTM
~~~~~~~~~~

.. autoclass:: Conv2DLSTM
   :members:

Conv3DLSTM
~~~~~~~~~~

.. autoclass:: Conv3DLSTM
   :members:

GRU
~~~

.. autoclass:: GRU
   :members:

Batch
-----

.. currentmodule:: sonnet

reshape
~~~~~~~

.. autofunction:: reshape

Reshape
~~~~~~~

.. autoclass:: Reshape
   :members:

flatten
~~~~~~~

.. autofunction:: flatten

Flatten
~~~~~~~

.. autoclass:: Flatten
   :members:

BatchApply
~~~~~~~~~~

.. autoclass:: BatchApply
   :members:

Embedding modules
-----------------

.. currentmodule:: sonnet

Embed
~~~~~

.. autoclass:: Embed
   :members:

Optimizers
----------

.. automodule:: sonnet.optimizers

Optimizer
~~~~~~~~~

.. autoclass:: Optimizer
   :members:

Adam
~~~~

.. autoclass:: Adam
   :members:

Momentum
~~~~~~~~

.. autoclass:: Momentum
   :members:

RMSProp
~~~~~~~

.. autoclass:: RMSProp
   :members:

SGD
~~~

.. autoclass:: SGD
   :members:

Initializers
------------

.. automodule:: sonnet.initializers

Initializer
~~~~~~~~~~~

.. autoclass:: Initializer
   :members:

Constant
~~~~~~~~

.. autoclass:: Constant
   :members:

Identity
~~~~~~~~

.. autoclass:: Identity
   :members:

Ones
~~~~

.. autoclass:: Ones
   :members:

Orthogonal
~~~~~~~~~~

.. autoclass:: Orthogonal
   :members:

RandomNormal
~~~~~~~~~~~~

.. autoclass:: RandomNormal
   :members:

RandomUniform
~~~~~~~~~~~~~

.. autoclass:: RandomUniform
   :members:

TruncatedNormal
~~~~~~~~~~~~~~~

.. autoclass:: TruncatedNormal
   :members:

VarianceScaling
~~~~~~~~~~~~~~~

.. autoclass:: VarianceScaling
   :members:

Zeros
~~~~~

.. autoclass:: Zeros
   :members:

Regularizers
------------

.. automodule:: sonnet.regularizers

Regularizer
~~~~~~~~~~~

.. autoclass:: Regularizer
   :members:

L1
~~

.. autoclass:: L1
   :members:

L2
~~

.. autoclass:: L2
   :members:

OffDiagonalOrthogonal
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: OffDiagonalOrthogonal
   :members:

Paddings
--------

.. automodule:: sonnet.pad

causal
~~~~~~

.. autofunction:: causal

create
~~~~~~

.. autofunction:: create

full
~~~~

.. autofunction:: full

reverse_causal
~~~~~~~~~~~~~~

.. autofunction:: reverse_causal

same
~~~~

.. autofunction:: same

valid
~~~~~

.. autofunction:: valid

.. TODO(petebu): better title?

Distribution
------------

.. automodule:: sonnet.distribute

Replicator
~~~~~~~~~~

.. autoclass:: Replicator
   :members:

TpuReplicator
~~~~~~~~~~~~~

.. autoclass:: TpuReplicator
   :members:

Metrics
-------

.. currentmodule:: sonnet

Metric
~~~~~~

.. autoclass:: Metric
   :members:

Mean
~~~~~~

.. autoclass:: Mean
   :members:

Sum
~~~~~~

.. autoclass:: Sum
   :members:

.. TODO(tomhennigan): rename to something more appropriate.

Nets
----

.. automodule:: sonnet.nets

MLP
~~~

.. autoclass:: MLP
   :members:

Cifar10ConvNet
~~~~~~~~~~~~~~

.. autoclass:: Cifar10ConvNet
   :members:

ResNet
~~~~~~~~~~~~~~

.. autoclass:: ResNet
   :members:

ResNet50
~~~~~~~~~~~~~~

.. autoclass:: ResNet50
   :members:

VectorQuantizer
~~~~~~~~~~~~~~~

.. autoclass:: VectorQuantizer
   :members:

VectorQuantizerEMA
~~~~~~~~~~~~~~~~~~

.. autoclass:: VectorQuantizerEMA
   :members:

Mixed Precision
---------------

.. automodule:: sonnet.mixed_precision

modes
~~~~~

.. autofunction:: modes

enable
~~~~~~

.. autofunction:: enable

disable
~~~~~~~

.. autofunction:: disable

scope
~~~~~

.. autofunction:: scope

References
----------

.. bibliography:: references.bib
   :style: unsrt
