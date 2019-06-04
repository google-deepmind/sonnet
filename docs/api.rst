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

Embedding modules
-----------------

.. currentmodule:: sonnet

Embed
~~~~~

.. autoclass:: Embed
   :members:

Normalization modules
---------------------

.. currentmodule:: sonnet

AxisNorm
~~~~~~~~

.. autoclass:: AxisNorm
   :members:

InstanceNorm
~~~~~~~~~~~~

.. autoclass:: InstanceNorm
   :members:

LayerNorm
~~~~~~~~~

.. autoclass:: LayerNorm
   :members:

BaseBatchNorm
~~~~~~~~~~~~~

.. autoclass:: BaseBatchNorm
   :members:

BatchNorm
~~~~~~~~~

.. autoclass:: BatchNorm
   :members:

GroupNorm
~~~~~~~~~

.. autoclass:: GroupNorm
   :members:

Recurrent modules
-----------------

.. currentmodule:: sonnet

RNNCore
~~~~~~~

.. autoclass:: RNNCore
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

ConvNDLSTM
~~~~~~~~~~

.. autoclass:: Conv1DLSTM
   :members:

.. autoclass:: Conv2DLSTM
   :members:

.. autoclass:: Conv3DLSTM
   :members:

GRU
~~~

.. autoclass:: GRU
   :members:

Utilities
---------

.. currentmodule:: sonnet

.. TODO(slebedev): move these to a more specific "topic"?

Metric
~~~~~~

.. autoclass:: Metric
   :members:

Reshape
~~~~~~~

.. autoclass:: Reshape
   :members:

Flatten
~~~~~~~

.. autoclass:: Flatten
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

Optimizers
----------

.. automodule:: sonnet.optimizers

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

Distribute
----------

.. automodule:: sonnet.distribute

Replicator
~~~~~~~~~~

.. autoclass:: Replicator
   :members:

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

References
----------

.. bibliography:: references.bib
   :style: unsrt
