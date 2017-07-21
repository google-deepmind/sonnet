# Copyright 2017 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""LSTM based modules for TensorFlow snt.

This python module contains LSTM-like cores that fall under the broader group
of RNN cores.  In general, initializers for the gate weights and other
model parameters may be passed to the constructor.

Typical usage example of the standard LSTM without peephole connections:

  ```
  import sonnet as snt

  hidden_size = 10
  batch_size = 2

  # Simple LSTM op on some input
  rnn = snt.LSTM(hidden_size)
  input = tf.placeholder(tf.float32, shape=[batch_size, hidden_size])
  out, next_state = rnn(input, rnn.initial_state())
  ```
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

# Dependency imports

from six.moves import xrange  # pylint: disable=redefined-builtin
from sonnet.python.modules import base
from sonnet.python.modules import basic
from sonnet.python.modules import batch_norm
from sonnet.python.modules import conv
from sonnet.python.modules import layer_norm
from sonnet.python.modules import rnn_core
from sonnet.python.modules import util
import tensorflow as tf

from tensorflow.python.ops import array_ops
from tensorflow.python.util import deprecation


class _BaseLSTM(rnn_core.RNNCore):
  """Base implementation underlying BatchNormLSTM and LSTM.

  This exists as a temporary measure during the deprecation process of batch
  norm options to LSTM. Once deprecation is complete, it will be removed and
  both BatchNormLSTM and LSTM replaced with simpler direct implementations.
  """
  # Keys that may be provided for parameter initializers.
  W_GATES = "w_gates"  # weight for gates
  B_GATES = "b_gates"  # bias of gates
  W_F_DIAG = "w_f_diag"  # weight for prev_cell -> forget gate peephole
  W_I_DIAG = "w_i_diag"  # weight for prev_cell -> input gate peephole
  W_O_DIAG = "w_o_diag"  # weight for prev_cell -> output gate peephole
  GAMMA_H = "gamma_h"  # batch norm scaling for previous_hidden -> gates
  GAMMA_X = "gamma_x"  # batch norm scaling for input -> gates
  GAMMA_C = "gamma_c"  # batch norm scaling for cell -> output
  BETA_C = "beta_c"  # (batch norm) bias for cell -> output
  POSSIBLE_INITIALIZER_KEYS = {W_GATES, B_GATES, W_F_DIAG, W_I_DIAG, W_O_DIAG,
                               GAMMA_H, GAMMA_X, GAMMA_C, BETA_C}
  # Keep old name for backwards compatibility

  POSSIBLE_KEYS = POSSIBLE_INITIALIZER_KEYS

  def __init__(self,
               hidden_size,
               forget_bias=1.0,
               initializers=None,
               partitioners=None,
               regularizers=None,
               use_peepholes=False,
               use_batch_norm_h=False,
               use_batch_norm_x=False,
               use_batch_norm_c=False,
               use_layer_norm=False,
               max_unique_stats=1,
               hidden_clip_value=None,
               cell_clip_value=None,
               name="lstm"):
    """See `__init__` of `LSTM` and `BatchNormLSTM` for docs."""
    super(_BaseLSTM, self).__init__(name=name)

    self._hidden_size = hidden_size
    self._forget_bias = forget_bias
    self._use_peepholes = use_peepholes
    self._max_unique_stats = max_unique_stats
    self._use_batch_norm_h = use_batch_norm_h
    self._use_batch_norm_x = use_batch_norm_x
    self._use_batch_norm_c = use_batch_norm_c
    self._use_layer_norm = use_layer_norm
    self._hidden_clip_value = hidden_clip_value
    self._cell_clip_value = cell_clip_value
    self.possible_keys = self.get_possible_initializer_keys(
        use_peepholes=use_peepholes, use_batch_norm_h=use_batch_norm_h,
        use_batch_norm_x=use_batch_norm_x, use_batch_norm_c=use_batch_norm_c)
    self._initializers = util.check_initializers(initializers,
                                                 self.possible_keys)
    self._partitioners = util.check_initializers(partitioners,
                                                 self.possible_keys)
    self._regularizers = util.check_initializers(regularizers,
                                                 self.possible_keys)
    if max_unique_stats < 1:
      raise ValueError("max_unique_stats must be >= 1")
    if max_unique_stats != 1 and not (
        use_batch_norm_h or use_batch_norm_x or use_batch_norm_c):
      raise ValueError("max_unique_stats specified but batch norm disabled")
    if use_batch_norm_h and use_layer_norm:
      raise ValueError(
          "Only one of use_batch_norm_h and layer_norm is allowed.")
    if use_batch_norm_x and use_layer_norm:
      raise ValueError(
          "Only one of use_batch_norm_x and layer_norm is allowed.")
    if use_batch_norm_c and use_layer_norm:
      raise ValueError(
          "Only one of use_batch_norm_c and layer_norm is allowed.")
    if hidden_clip_value is not None and hidden_clip_value < 0:
      raise ValueError("The value of hidden_clip_value should be nonnegative.")
    if cell_clip_value is not None and cell_clip_value < 0:
      raise ValueError("The value of cell_clip_value should be nonnegative.")

    if use_batch_norm_h:
      self._batch_norm_h = _BaseLSTM.IndexedStatsBatchNorm(max_unique_stats,
                                                           "batch_norm_h")
    if use_batch_norm_x:
      self._batch_norm_x = _BaseLSTM.IndexedStatsBatchNorm(max_unique_stats,
                                                           "batch_norm_x")
    if use_batch_norm_c:
      self._batch_norm_c = _BaseLSTM.IndexedStatsBatchNorm(max_unique_stats,
                                                           "batch_norm_c")

  def with_batch_norm_control(self, is_training, test_local_stats=True):
    """Wraps this RNNCore with the additional control input to the `BatchNorm`s.

    Example usage:

      lstm = snt.BatchNormLSTM(4)
      is_training = tf.placeholder(tf.bool)
      rnn_input = ...
      my_rnn = rnn.rnn(lstm.with_batch_norm_control(is_training), rnn_input)

    Args:
      is_training: Boolean that indicates whether we are in
        training mode or testing mode. When in training mode, the batch norm
        statistics are taken from the given batch, and moving statistics are
        updated. When in testing mode, the moving statistics are not updated,
        and in addition if `test_local_stats` is False then the moving
        statistics are used for the batch statistics. See the `BatchNorm` module
        for more details.
      test_local_stats: Boolean scalar indicated whether to use local
        batch statistics in test mode.

    Returns:
      RNNCell wrapping this class with the extra input(s) added.
    """
    return _BaseLSTM.CellWithExtraInput(self,
                                        is_training=is_training,
                                        test_local_stats=test_local_stats)

  @classmethod
  def get_possible_initializer_keys(
      cls, use_peepholes=False, use_batch_norm_h=False, use_batch_norm_x=False,
      use_batch_norm_c=False):
    """Returns the keys the dictionary of variable initializers may contain.

    The set of all possible initializer keys are:
      w_gates:  weight for gates
      b_gates:  bias of gates
      w_f_diag: weight for prev_cell -> forget gate peephole
      w_i_diag: weight for prev_cell -> input gate peephole
      w_o_diag: weight for prev_cell -> output gate peephole
      gamma_h:  batch norm scaling for previous_hidden -> gates
      gamma_x:  batch norm scaling for input -> gates
      gamma_c:  batch norm scaling for cell -> output
      beta_c:   batch norm bias for cell -> output

    Args:
      cls:The class.
      use_peepholes: Boolean that indicates whether peephole connections are
        used.
      use_batch_norm_h: Boolean that indicates whether to apply batch
        normalization at the previous_hidden -> gates contribution. If you are
        experimenting with batch norm then this may be the most effective to
        turn on.
      use_batch_norm_x: Boolean that indicates whether to apply batch
        normalization at the input -> gates contribution.
      use_batch_norm_c: Boolean that indicates whether to apply batch
        normalization at the cell -> output contribution.

    Returns:
      Set with strings corresponding to the strings that may be passed to the
        constructor.
    """

    possible_keys = cls.POSSIBLE_INITIALIZER_KEYS.copy()
    if not use_peepholes:
      possible_keys.difference_update(
          {cls.W_F_DIAG, cls.W_I_DIAG, cls.W_O_DIAG})
    if not use_batch_norm_h:
      possible_keys.remove(cls.GAMMA_H)
    if not use_batch_norm_x:
      possible_keys.remove(cls.GAMMA_X)
    if not use_batch_norm_c:
      possible_keys.difference_update({cls.GAMMA_C, cls.BETA_C})
    return possible_keys

  def _build(self, inputs, prev_state, is_training=None, test_local_stats=True):
    """Connects the LSTM module into the graph.

    If this is not the first time the module has been connected to the graph,
    the Tensors provided as inputs and state must have the same final
    dimension, in order for the existing variables to be the correct size for
    their corresponding multiplications. The batch size may differ for each
    connection.

    Args:
      inputs: Tensor of size `[batch_size, input_size]`.
      prev_state: Tuple (prev_hidden, prev_cell), or if batch norm is enabled
        and `max_unique_stats > 1`, then (prev_hidden, prev_cell, time_step).
        Here, prev_hidden and prev_cell are tensors of size
        `[batch_size, hidden_size]`, and time_step is used to indicate the
        current RNN step.
      is_training: Boolean indicating whether we are in training mode (as
        opposed to testing mode), passed to the batch norm
        modules. Note to use this you must wrap the cell via the
        `with_batch_norm_control` function.
      test_local_stats: Boolean indicating whether to use local batch statistics
        in test mode. See the `BatchNorm` documentation for more on this.

    Returns:
      A tuple (output, next_state) where 'output' is a Tensor of size
      `[batch_size, hidden_size]` and 'next_state' is a tuple
      (next_hidden, next_cell) or (next_hidden, next_cell, time_step + 1),
      where next_hidden and next_cell have size `[batch_size, hidden_size]`.

    Raises:
      ValueError: If connecting the module into the graph any time after the
        first time, and the inferred size of the inputs does not match previous
        invocations.
    """
    use_batch_norm = self._use_batch_norm_c or self._use_batch_norm_h
    use_batch_norm = use_batch_norm or self._use_batch_norm_x
    if use_batch_norm and is_training is None:
      raise ValueError("Boolean is_training flag must be explicitly specified "
                       "when using batch normalization.")

    if self._max_unique_stats == 1:
      prev_hidden, prev_cell = prev_state
      time_step = None
    else:
      prev_hidden, prev_cell, time_step = prev_state

    # pylint: disable=invalid-unary-operand-type
    if self._hidden_clip_value is not None:
      prev_hidden = tf.clip_by_value(
          prev_hidden, -self._hidden_clip_value, self._hidden_clip_value)
    if self._cell_clip_value is not None:
      prev_cell = tf.clip_by_value(
          prev_cell, -self._cell_clip_value, self._cell_clip_value)
    # pylint: enable=invalid-unary-operand-type

    self._create_gate_variables(inputs.get_shape(), inputs.dtype)
    self._create_batch_norm_variables(inputs.dtype)

    # pylint false positive: calling module of same file;
    # pylint: disable=not-callable

    if self._use_batch_norm_h or self._use_batch_norm_x:
      gates_h = tf.matmul(prev_hidden, self._w_h)
      gates_x = tf.matmul(inputs, self._w_x)
      if self._use_batch_norm_h:
        gates_h = self._gamma_h * self._batch_norm_h(gates_h,
                                                     time_step,
                                                     is_training,
                                                     test_local_stats)
      if self._use_batch_norm_x:
        gates_x = self._gamma_x * self._batch_norm_x(gates_x,
                                                     time_step,
                                                     is_training,
                                                     test_local_stats)
      gates = gates_h + gates_x
    else:
      # Parameters of gates are concatenated into one multiply for efficiency.
      inputs_and_hidden = tf.concat([inputs, prev_hidden], 1)
      gates = tf.matmul(inputs_and_hidden, self._w_xh)

      if self._use_layer_norm:
        gates = layer_norm.LayerNorm()(gates)

    gates += self._b

    # i = input_gate, j = next_input, f = forget_gate, o = output_gate
    i, j, f, o = array_ops.split(value=gates, num_or_size_splits=4, axis=1)

    if self._use_peepholes:  # diagonal connections
      self._create_peephole_variables(inputs.dtype)
      f += self._w_f_diag * prev_cell
      i += self._w_i_diag * prev_cell

    forget_mask = tf.sigmoid(f + self._forget_bias)
    next_cell = forget_mask * prev_cell + tf.sigmoid(i) * tf.tanh(j)
    cell_output = next_cell
    if self._use_batch_norm_c:
      cell_output = (self._beta_c
                     + self._gamma_c * self._batch_norm_c(cell_output,
                                                          time_step,
                                                          is_training,
                                                          test_local_stats))
    if self._use_peepholes:
      cell_output += self._w_o_diag * cell_output
    next_hidden = tf.tanh(cell_output) * tf.sigmoid(o)

    if self._max_unique_stats == 1:
      return next_hidden, (next_hidden, next_cell)
    else:
      return next_hidden, (next_hidden, next_cell, time_step + 1)

  def _create_batch_norm_variables(self, dtype):
    """Initialize the variables used for the `BatchNorm`s (if any)."""
    # The paper recommends a value of 0.1 for good gradient flow through the
    # tanh nonlinearity (although doesn't say whether this is for all gammas,
    # or just some).
    gamma_initializer = tf.constant_initializer(0.1)

    if self._use_batch_norm_h:
      self._gamma_h = tf.get_variable(
          self.GAMMA_H,
          shape=[4 * self._hidden_size],
          dtype=dtype,
          initializer=self._initializers.get(self.GAMMA_H, gamma_initializer),
          partitioner=self._partitioners.get(self.GAMMA_H),
          regularizer=self._regularizers.get(self.GAMMA_H))
    if self._use_batch_norm_x:
      self._gamma_x = tf.get_variable(
          self.GAMMA_X,
          shape=[4 * self._hidden_size],
          dtype=dtype,
          initializer=self._initializers.get(self.GAMMA_X, gamma_initializer),
          partitioner=self._partitioners.get(self.GAMMA_X),
          regularizer=self._regularizers.get(self.GAMMA_X))
    if self._use_batch_norm_c:
      self._gamma_c = tf.get_variable(
          self.GAMMA_C,
          shape=[self._hidden_size],
          dtype=dtype,
          initializer=self._initializers.get(self.GAMMA_C, gamma_initializer),
          partitioner=self._partitioners.get(self.GAMMA_C),
          regularizer=self._regularizers.get(self.GAMMA_C))
      self._beta_c = tf.get_variable(
          self.BETA_C,
          shape=[self._hidden_size],
          dtype=dtype,
          initializer=self._initializers.get(self.BETA_C),
          partitioner=self._partitioners.get(self.BETA_C),
          regularizer=self._regularizers.get(self.BETA_C))

  def _create_gate_variables(self, input_shape, dtype):
    """Initialize the variables used for the gates."""
    if len(input_shape) != 2:
      raise ValueError(
          "Rank of shape must be {} not: {}".format(2, len(input_shape)))
    input_size = input_shape.dims[1].value

    b_shape = [4 * self._hidden_size]

    equiv_input_size = self._hidden_size + input_size
    initializer = basic.create_linear_initializer(equiv_input_size)

    if self._use_batch_norm_h or self._use_batch_norm_x:
      self._w_h = tf.get_variable(
          self.W_GATES + "_H",
          shape=[self._hidden_size, 4 * self._hidden_size],
          dtype=dtype,
          initializer=self._initializers.get(self.W_GATES, initializer),
          partitioner=self._partitioners.get(self.W_GATES),
          regularizer=self._regularizers.get(self.W_GATES))
      self._w_x = tf.get_variable(
          self.W_GATES + "_X",
          shape=[input_size, 4 * self._hidden_size],
          dtype=dtype,
          initializer=self._initializers.get(self.W_GATES, initializer),
          partitioner=self._partitioners.get(self.W_GATES),
          regularizer=self._regularizers.get(self.W_GATES))
    else:
      self._w_xh = tf.get_variable(
          self.W_GATES,
          shape=[self._hidden_size + input_size, 4 * self._hidden_size],
          dtype=dtype,
          initializer=self._initializers.get(self.W_GATES, initializer),
          partitioner=self._partitioners.get(self.W_GATES),
          regularizer=self._regularizers.get(self.W_GATES))
    self._b = tf.get_variable(
        self.B_GATES,
        shape=b_shape,
        dtype=dtype,
        initializer=self._initializers.get(self.B_GATES, initializer),
        partitioner=self._partitioners.get(self.B_GATES),
        regularizer=self._regularizers.get(self.B_GATES))

  def _create_peephole_variables(self, dtype):
    """Initialize the variables used for the peephole connections."""
    self._w_f_diag = tf.get_variable(
        self.W_F_DIAG,
        shape=[self._hidden_size],
        dtype=dtype,
        initializer=self._initializers.get(self.W_F_DIAG),
        partitioner=self._partitioners.get(self.W_F_DIAG),
        regularizer=self._regularizers.get(self.W_F_DIAG))
    self._w_i_diag = tf.get_variable(
        self.W_I_DIAG,
        shape=[self._hidden_size],
        dtype=dtype,
        initializer=self._initializers.get(self.W_I_DIAG),
        partitioner=self._partitioners.get(self.W_I_DIAG),
        regularizer=self._regularizers.get(self.W_I_DIAG))
    self._w_o_diag = tf.get_variable(
        self.W_O_DIAG,
        shape=[self._hidden_size],
        dtype=dtype,
        initializer=self._initializers.get(self.W_O_DIAG),
        partitioner=self._partitioners.get(self.W_O_DIAG),
        regularizer=self._regularizers.get(self.W_O_DIAG))

  def initial_state(self, batch_size, dtype=tf.float32, trainable=False,
                    trainable_initializers=None, trainable_regularizers=None,
                    name=None):
    """Builds the default start state tensor of zeros.

    Args:
      batch_size: An int, float or scalar Tensor representing the batch size.
      dtype: The data type to use for the state.
      trainable: Boolean that indicates whether to learn the initial state.
      trainable_initializers: An optional pair of initializers for the
          initial hidden state and cell state.
      trainable_regularizers: Optional regularizer function or nested structure
        of functions with the same structure as the `state_size` property of the
        core, to be used as regularizers of the initial state variable. A
        regularizer should be a function that takes a single `Tensor` as an
        input and returns a scalar `Tensor` output, e.g. the L1 and L2
        regularizers in `tf.contrib.layers`.
      name: Optional string used to prefix the initial state variable names, in
          the case of a trainable initial state. If not provided, defaults to
          the name of the module.

    Returns:
      A tensor tuple `([batch_size, state_size], [batch_size, state_size], ?)`
      filled with zeros, with the third entry present when batch norm is enabled
      with `max_unique_stats > 1', with value `0` (representing the time step).
    """
    if self._max_unique_stats == 1:
      return super(_BaseLSTM, self).initial_state(
          batch_size, dtype=dtype, trainable=trainable,
          trainable_initializers=trainable_initializers,
          trainable_regularizers=trainable_regularizers, name=name)
    else:
      with tf.name_scope(self._initial_state_scope(name)):
        if not trainable:
          state = self.zero_state(batch_size, dtype)
        else:
          # We have to manually create the state ourselves so we don't create a
          # variable that never gets used for the third entry.
          state = rnn_core.trainable_initial_state(
              batch_size,
              (tf.TensorShape([self._hidden_size]),
               tf.TensorShape([self._hidden_size])),
              dtype=dtype,
              initializers=trainable_initializers,
              regularizers=trainable_regularizers,
              name=self._initial_state_scope(name))
        return (state[0], state[1], tf.constant(0, dtype=tf.int32))

  @property
  def state_size(self):
    """Tuple of `tf.TensorShape`s indicating the size of state tensors."""
    if self._max_unique_stats == 1:
      return (tf.TensorShape([self._hidden_size]),
              tf.TensorShape([self._hidden_size]))
    else:
      return (tf.TensorShape([self._hidden_size]),
              tf.TensorShape([self._hidden_size]),
              tf.TensorShape(1))

  @property
  def output_size(self):
    """`tf.TensorShape` indicating the size of the core output."""
    return tf.TensorShape([self._hidden_size])

  @property
  def use_peepholes(self):
    """Boolean indicating whether peephole connections are used."""
    return self._use_peepholes

  @property
  def use_batch_norm_h(self):
    """Boolean indicating whether batch norm for hidden -> gates is enabled."""
    return self._use_batch_norm_h

  @property
  def use_batch_norm_x(self):
    """Boolean indicating whether batch norm for input -> gates is enabled."""
    return self._use_batch_norm_x

  @property
  def use_batch_norm_c(self):
    """Boolean indicating whether batch norm for cell -> output is enabled."""
    return self._use_batch_norm_c

  @property
  def use_layer_norm(self):
    """Boolean indicating whether layer norm is enabled."""
    return self._use_layer_norm

  class IndexedStatsBatchNorm(base.AbstractModule):
    """BatchNorm module where batch statistics are selected by an input index.

    This is used by LSTM+batchnorm, where we have distinct batch norm statistics
    for the first `max_unique_stats` time steps, and then use the final set of
    statistics for subsequent time steps.

    The module has as input (x, index, is_training, test_local_stats). During
    training or when test_local_stats=True, the output is simply batchnorm(x)
    (where mean(x) and stddev(x) are used), and during training the
    `BatchNorm` module accumulates statistics in mean_i, etc, where
    i = min(index, max_unique_stats - 1).

    During testing with test_local_stats=False, the output is batchnorm(x),
    where mean_i and stddev_i are used instead of mean(x) and stddev(x).

    See the `BatchNorm` module for more on is_training and test_local_stats.

    No offset `beta` or scaling `gamma` are learnt.
    """

    def __init__(self, max_unique_stats, name=None):
      """Create an IndexedStatsBatchNorm.

      Args:
        max_unique_stats: number of different indices to have statistics for;
          indices beyond this will use the final statistics.
        name: Name of the module.
      """
      super(_BaseLSTM.IndexedStatsBatchNorm, self).__init__(name=name)
      self._max_unique_stats = max_unique_stats

    def _build(self, inputs, index, is_training, test_local_stats):
      """Add the IndexedStatsBatchNorm module to the graph.

      Args:
        inputs: Tensor to apply batch norm to.
        index: Scalar TensorFlow int32 value to select the batch norm index.
        is_training: Boolean to indicate to `snt.BatchNorm` if we are
          currently training.
        test_local_stats: Boolean to indicate to `snt.BatchNorm` if batch
          normalization should  use local batch statistics at test time.

      Returns:
        Output of batch norm operation.
      """
      def create_batch_norm():
        return batch_norm.BatchNorm(offset=False, scale=False)(
            inputs, is_training, test_local_stats)

      if self._max_unique_stats > 1:
        pred_fn_pairs = [(tf.equal(i, index), create_batch_norm)
                         for i in xrange(self._max_unique_stats - 1)]
        out = tf.case(pred_fn_pairs, create_batch_norm)
        out.set_shape(inputs.get_shape())  # needed for tf.case shape inference
        return out
      else:
        return create_batch_norm()

  class CellWithExtraInput(tf.contrib.rnn.RNNCell):
    """Wraps an RNNCell to create a new RNNCell with extra input appended.

    This will pass the additional input `args` and `kwargs` to the __call__
    function of the RNNCell after the input and prev_state inputs.
    """

    def __init__(self, cell, *args, **kwargs):
      """Construct the CellWithExtraInput.

      Args:
        cell: The RNNCell to wrap (typically a snt.RNNCore).
        *args: Extra arguments to pass to __call__.
        **kwargs: Extra keyword arguments to pass to __call__.
      """
      self._cell = cell
      self._args = args
      self._kwargs = kwargs

    def __call__(self, inputs, state):
      return self._cell(inputs, state, *self._args, **self._kwargs)

    @property
    def state_size(self):
      """Tuple indicating the size of nested state tensors."""
      return self._cell.state_size

    @property
    def output_size(self):
      """`tf.TensorShape` indicating the size of the core output."""
      return self._cell.output_size


class LSTM(_BaseLSTM):
  """LSTM recurrent network cell with optional peepholes & layer normalization.

  The implementation is based on: http://arxiv.org/abs/1409.2329. We add
  forget_bias (default: 1) to the biases of the forget gate in order to
  reduce the scale of forgetting in the beginning of the training.

  #### Layer normalization

  This is described in https://arxiv.org/pdf/1607.06450.pdf

  #### Peep-hole connections

  Peep-hole connections may optionally be used by specifying a flag in the
  constructor. These connections can aid increasing the precision of output
  timing, for more details see:

    https://research.google.com/pubs/archive/43905.pdf

  Attributes:
    state_size: Tuple of `tf.TensorShape`s indicating the size of state tensors.
    output_size: `tf.TensorShape` indicating the size of the core output.
    use_peepholes: Boolean indicating whether peephole connections are used.
  """

  @deprecation.deprecated_args(
      "2017-09-18",
      "Please switch from LSTM to BatchNormLSTM if you need batch norm "
      "functionality.",
      "use_batch_norm_h", "use_batch_norm_x", "use_batch_norm_c",
      "max_unique_stats")
  def __init__(self,
               hidden_size,
               forget_bias=1.0,
               initializers=None,
               partitioners=None,
               regularizers=None,
               use_peepholes=False,
               use_batch_norm_h=False,
               use_batch_norm_x=False,
               use_batch_norm_c=False,
               use_layer_norm=False,
               max_unique_stats=1,
               hidden_clip_value=None,
               cell_clip_value=None,
               name="lstm"):
    """Construct LSTM.

    Args:
      hidden_size: (int) Hidden size dimensionality.
      forget_bias: (float) Bias for the forget activation.
      initializers: Dict containing ops to initialize the weights.
        This dictionary may contain any of the keys returned by
        `LSTM.get_possible_initializer_keys`.
        The gamma and beta variables control batch normalization values for
        different batch norm transformations inside the cell; see the paper for
        details.
      partitioners: Optional dict containing partitioners to partition
        the weights and biases. As a default, no partitioners are used. This
        dict may contain any of the keys returned by
        `LSTM.get_possible_initializer_keys`.
      regularizers: Optional dict containing regularizers for the weights and
        biases. As a default, no regularizers are used. This dict may contain
        any of the keys returned by
        `LSTM.get_possible_initializer_keys`.
      use_peepholes: Boolean that indicates whether peephole connections are
        used.
      use_batch_norm_h: Boolean that indicates whether to apply batch
        normalization at the previous_hidden -> gates contribution.
        This is deprecated and will be removed in a future sonnet version.
        Please switch to `BatchNormLSTM` if you require it.
      use_batch_norm_x: Boolean that indicates whether to apply batch
        normalization at the input -> gates contribution.
        This is deprecated and will be removed in a future sonnet version.
        Please switch to `BatchNormLSTM` if you require it.
      use_batch_norm_c: Boolean that indicates whether to apply batch
        normalization at the cell -> output contribution.
        This is deprecated and will be removed in a future sonnet version.
        Please switch to `BatchNormLSTM` if you require it.
      use_layer_norm: Boolean that indicates whether to apply layer
        normalization.
      max_unique_stats: The maximum number of steps to use unique batch norm
        statistics for. (See module description above for more details.)
        This is deprecated and will be removed in a future sonnet version.
        Please switch to `BatchNormLSTM` if you require it.
      hidden_clip_value: Optional number; if set, then the LSTM hidden state
        vector is clipped by this value.
      cell_clip_value: Optional number; if set, then the LSTM cell vector is
        clipped by this value.
      name: name of the module.

    Raises:
      KeyError: if `initializers` contains any keys not returned by
        `LSTM.get_possible_initializer_keys`.
      KeyError: if `partitioners` contains any keys not returned by
        `LSTM.get_possible_initializer_keys`.
      KeyError: if `regularizers` contains any keys not returned by
        `LSTM.get_possible_initializer_keys`.
      ValueError: if a peephole initializer is passed in the initializer list,
        but `use_peepholes` is False.
    """
    super(LSTM, self).__init__(
        hidden_size,
        forget_bias=forget_bias,
        initializers=initializers,
        partitioners=partitioners,
        regularizers=regularizers,
        use_peepholes=use_peepholes,
        use_batch_norm_h=use_batch_norm_h,
        use_batch_norm_x=use_batch_norm_x,
        use_batch_norm_c=use_batch_norm_c,
        use_layer_norm=use_layer_norm,
        max_unique_stats=max_unique_stats,
        hidden_clip_value=hidden_clip_value,
        cell_clip_value=cell_clip_value,
        name=name)


class BatchNormLSTM(_BaseLSTM):
  """LSTM recurrent network cell with optional peepholes, batch normalization.

  The base implementation is based on: http://arxiv.org/abs/1409.2329. We add
  forget_bias (default: 1) to the biases of the forget gate in order to
  reduce the scale of forgetting in the beginning of the training.

  #### Peep-hole connections

  Peep-hole connections may optionally be used by specifying a flag in the
  constructor. These connections can aid increasing the precision of output
  timing, for more details see:

    https://research.google.com/pubs/archive/43905.pdf

  #### Batch normalization

  The batch norm transformation (in training mode) is
    batchnorm(x) = gamma * (x - mean(x)) / stddev(x) + beta,
  where gamma is a learnt scaling factor and beta is a learnt offset.

  Batch normalization may optionally be used at different places in the LSTM by
  specifying flag(s) in the constructor. These are applied when calculating
  the gate activations and cell-to-hidden transformation. The set-up is based on

    https://arxiv.org/pdf/1603.09025.pdf

  ##### Batch normalization: where to apply?

  Batch norm can be applied in three different places in the LSTM:

    (h) To the W_h h_{t-1} contribution to the gates from the previous hiddens.
    (x) To the W_x x_t contribution to the gates from the current input.
    (c) To the cell value c_t when calculating the output h_t from the cell.

  (The notation here is consistent with the Recurrent Batch Normalization
  paper). Each of these can be controlled individually, because batch norm is
  expensive, and not all are necessary. The paper doesn't mention the relative
  effects of these different batch norms; however, experimentation with a
  shallow LSTM for the `permuted_mnist` sequence task suggests that (h) is the
  most important and the other two can be left off. For other tasks or deeper
  (stacked) LSTMs, other batch norm combinations may be more effective.

  ##### Batch normalization: collecting stats (training vs test)

  When switching to testing (see `LSTM.with_batch_norm_control`), we can use a
  mean and stddev learnt from the training data instead of using the statistics
  from the test data. (This both increases test accuracy because the statistics
  have less variance, and if the test data does not have the same distribution
  as the training data then we must use the training statistics to ensure the
  effective network does not change when switching to testing anyhow.)

  This does however introduces a slight subtlety. The first few time steps of
  the RNN tend to have varying statistics (mean and variance) before settling
  down to a steady value. Therefore in general, better performance is obtained
  by using separate statistics for the first few time steps, and then using the
  final set of statistics for all subsequent time steps. This is controlled by
  the parameter `max_unique_stats`. (We can't have an unbounded number of
  distinct statistics for both technical reasons and also for the case where
  test sequences are longer than anything seen in training.)

  You may be fine leaving it at its default value of 1. Small values (like 10)
  may achieve better performance on some tasks when testing with cached
  statistics.

  Attributes:
    state_size: Tuple of `tf.TensorShape`s indicating the size of state tensors.
    output_size: `tf.TensorShape` indicating the size of the core output.
    use_peepholes: Boolean indicating whether peephole connections are used.
    use_batch_norm_h: Boolean indicating whether batch norm (h) is enabled.
    use_batch_norm_x: Boolean indicating whether batch norm (x) is enabled.
    use_batch_norm_c: Boolean indicating whether batch norm (c) is enabled.
  """

  def __init__(self,
               hidden_size,
               forget_bias=1.0,
               initializers=None,
               partitioners=None,
               regularizers=None,
               use_peepholes=False,
               use_batch_norm_h=True,
               use_batch_norm_x=False,
               use_batch_norm_c=False,
               max_unique_stats=1,
               hidden_clip_value=None,
               cell_clip_value=None,
               name="batch_norm_lstm"):
    """Construct `BatchNormLSTM`.

    Args:
      hidden_size: (int) Hidden size dimensionality.
      forget_bias: (float) Bias for the forget activation.
      initializers: Dict containing ops to initialize the weights.
        This dictionary may contain any of the keys returned by
        `BatchNormLSTM.get_possible_initializer_keys`.
        The gamma and beta variables control batch normalization values for
        different batch norm transformations inside the cell; see the paper for
        details.
      partitioners: Optional dict containing partitioners to partition
        the weights and biases. As a default, no partitioners are used. This
        dict may contain any of the keys returned by
        `BatchNormLSTM.get_possible_initializer_keys`.
      regularizers: Optional dict containing regularizers for the weights and
        biases. As a default, no regularizers are used. This dict may contain
        any of the keys returned by
        `BatchNormLSTM.get_possible_initializer_keys`.
      use_peepholes: Boolean that indicates whether peephole connections are
        used.
      use_batch_norm_h: Boolean that indicates whether to apply batch
        normalization at the previous_hidden -> gates contribution. If you are
        experimenting with batch norm then this may be the most effective to
        use, and is enabled by default.
      use_batch_norm_x: Boolean that indicates whether to apply batch
        normalization at the input -> gates contribution.
      use_batch_norm_c: Boolean that indicates whether to apply batch
        normalization at the cell -> output contribution.
      max_unique_stats: The maximum number of steps to use unique batch norm
        statistics for. (See module description above for more details.)
      hidden_clip_value: Optional number; if set, then the LSTM hidden state
        vector is clipped by this value.
      cell_clip_value: Optional number; if set, then the LSTM cell vector is
        clipped by this value.
      name: name of the module.

    Raises:
      KeyError: if `initializers` contains any keys not returned by
        `BatchNormLSTM.get_possible_initializer_keys`.
      KeyError: if `partitioners` contains any keys not returned by
        `BatchNormLSTM.get_possible_initializer_keys`.
      KeyError: if `regularizers` contains any keys not returned by
        `BatchNormLSTM.get_possible_initializer_keys`.
      ValueError: if a peephole initializer is passed in the initializer list,
        but `use_peepholes` is False.
      ValueError: if a batch norm initializer is passed in the initializer list,
        but batch norm is disabled.
      ValueError: if none of the `use_batch_norm_*` options are True.
      ValueError: if `max_unique_stats` is < 1.
    """
    if not any([use_batch_norm_h, use_batch_norm_x, use_batch_norm_c]):
      raise ValueError("At least one use_batch_norm_* option is required for "
                       "BatchNormLSTM")
    super(BatchNormLSTM, self).__init__(
        hidden_size,
        forget_bias=forget_bias,
        initializers=initializers,
        partitioners=partitioners,
        regularizers=regularizers,
        use_peepholes=use_peepholes,
        use_batch_norm_h=use_batch_norm_h,
        use_batch_norm_x=use_batch_norm_x,
        use_batch_norm_c=use_batch_norm_c,
        max_unique_stats=max_unique_stats,
        hidden_clip_value=hidden_clip_value,
        cell_clip_value=cell_clip_value,
        name=name)

  # Overriding because the default for use_batch_norm_h is True here.
  @classmethod
  def get_possible_initializer_keys(
      cls, use_peepholes=False, use_batch_norm_h=True, use_batch_norm_x=False,
      use_batch_norm_c=False):
    return super(BatchNormLSTM, cls).get_possible_initializer_keys(
        use_peepholes=use_peepholes, use_batch_norm_h=use_batch_norm_h,
        use_batch_norm_x=use_batch_norm_x, use_batch_norm_c=use_batch_norm_c)


class ConvLSTM(rnn_core.RNNCore):
  """Convolutional LSTM."""

  @classmethod
  def get_possible_initializer_keys(cls, conv_ndims, use_bias=True):
    conv_class = cls._get_conv_class(conv_ndims)
    return conv_class.get_possible_initializer_keys(use_bias)

  @classmethod
  def _get_conv_class(cls, conv_ndims):
    if conv_ndims == 1:
      return conv.Conv1D
    elif conv_ndims == 2:
      return conv.Conv2D
    elif conv_ndims == 3:
      return conv.Conv3D
    else:
      raise ValueError("Invalid convolution dimensionality.")

  def __init__(self,
               conv_ndims,
               input_shape,
               output_channels,
               kernel_shape,
               stride=1,
               padding=conv.SAME,
               use_bias=True,
               skip_connection=False,
               forget_bias=1.0,
               initializers=None,
               partitioners=None,
               regularizers=None,
               name="conv_lstm"):
    """Construct ConvLSTM.

    Args:
      conv_ndims: Convolution dimensionality (1, 2 or 3).
      input_shape: Shape of the input as tuple, excluding the batch size.
      output_channels: Number of output channels of the conv LSTM.
      kernel_shape: Sequence of kernel sizes (of size 2), or integer that is
          used to define kernel size in all dimensions.
      stride: Sequence of kernel strides (of size 2), or integer that is used to
          define stride in all dimensions.
      padding: Padding algorithm, either `snt.SAME` or `snt.VALID`.
      use_bias: Use bias in convolutions.
      skip_connection: If set to `True`, concatenate the input to the output
          of the conv LSTM. Default: `False`.
      forget_bias: Forget bias.
      initializers: Dict containing ops to initialize the convolutional weights.
      partitioners: Optional dict containing partitioners to partition
        the convolutional weights and biases. As a default, no partitioners are
        used.
      regularizers: Optional dict containing regularizers for the convolutional
        weights and biases. As a default, no regularizers are used.
      name: Name of the module.

    Raises:
      ValueError: If `skip_connection` is `True` and stride is different from 1
        or if `input_shape` is incompatible with `conv_ndims`.
    """
    super(ConvLSTM, self).__init__(name=name)

    self._conv_class = self._get_conv_class(conv_ndims)

    if skip_connection and stride != 1:
      raise ValueError("`stride` needs to be 1 when using skip connection")

    if conv_ndims != len(input_shape)-1:
      raise ValueError("Invalid input_shape {} for conv_ndims={}.".format(
          input_shape, conv_ndims))

    self._conv_ndims = conv_ndims
    self._input_shape = input_shape
    self._output_channels = output_channels
    self._kernel_shape = kernel_shape
    self._stride = stride
    self._padding = padding
    self._use_bias = use_bias
    self._forget_bias = forget_bias
    self._skip_connection = skip_connection
    self._initializers = initializers
    self._partitioners = partitioners
    self._regularizers = regularizers

    self._total_output_channels = output_channels
    if self._stride != 1:
      self._total_output_channels //= self._stride * self._stride
    if self._skip_connection:
      self._total_output_channels += self._input_shape[-1]

    self._convolutions = collections.defaultdict(self._new_convolution)

  def _new_convolution(self):
    return self._conv_class(
        output_channels=4*self._output_channels,
        kernel_shape=self._kernel_shape,
        stride=self._stride,
        padding=self._padding,
        use_bias=self._use_bias,
        initializers=self._initializers,
        partitioners=self._partitioners,
        regularizers=self._regularizers,
        name="conv")

  @property
  def convolutions(self):
    return self._convolutions

  @property
  def state_size(self):
    """Tuple of `tf.TensorShape`s indicating the size of state tensors."""
    hidden_size = tf.TensorShape(self._input_shape[:-1] +
                                 (self._output_channels,))
    return (hidden_size, hidden_size)

  @property
  def output_size(self):
    """`tf.TensorShape` indicating the size of the core output."""
    return tf.TensorShape(self._input_shape[:-1] +
                          (self._total_output_channels,))

  def _build(self, inputs, state):
    hidden, cell = state
    input_conv = self._convolutions["input"]
    hidden_conv = self._convolutions["hidden"]
    next_hidden = input_conv(inputs) + hidden_conv(hidden)
    gates = tf.split(value=next_hidden, num_or_size_splits=4,
                     axis=self._conv_ndims+1)

    input_gate, next_input, forget_gate, output_gate = gates
    next_cell = tf.sigmoid(forget_gate + self._forget_bias) * cell
    next_cell += tf.sigmoid(input_gate) * tf.tanh(next_input)
    output = tf.tanh(next_cell) * tf.sigmoid(output_gate)

    if self._skip_connection:
      output = tf.concat([output, inputs], axis=-1)
    return output, (output, next_cell)


class Conv1DLSTM(ConvLSTM):
  """1D convolutional LSTM."""

  @classmethod
  def get_possible_initializer_keys(cls, use_bias=True):
    return super(Conv1DLSTM, cls).get_possible_initializer_keys(1, use_bias)

  def __init__(self, name="conv_1d_lstm", **kwargs):
    """Construct Conv1DLSTM. See `snt.ConvLSTM` for more details."""
    super(Conv1DLSTM, self).__init__(conv_ndims=1, name=name, **kwargs)


class Conv2DLSTM(ConvLSTM):
  """2D convolutional LSTM."""


  @classmethod
  def get_possible_initializer_keys(cls, use_bias=True):
    return super(Conv2DLSTM, cls).get_possible_initializer_keys(2, use_bias)

  def __init__(self, name="conv_2d_lstm", **kwargs):
    """Construct Conv2DLSTM. See `snt.ConvLSTM` for more details."""
    super(Conv2DLSTM, self).__init__(conv_ndims=2, name=name, **kwargs)


class GRU(rnn_core.RNNCore):
  """GRU recurrent network cell.

  The implementation is based on: https://arxiv.org/pdf/1412.3555v1.pdf.

  Attributes:
    state_size: Integer indicating the size of state tensor.
    output_size: Integer indicating the size of the core output.
  """

  # Keys that may be provided for parameter initializers.
  WZ = "wz"  # weight for input -> update cell
  UZ = "uz"  # weight for prev_state -> update cell
  BZ = "bz"  # bias for update_cell
  WR = "wr"  # weight for input -> reset cell
  UR = "ur"  # weight for prev_state -> reset cell
  BR = "br"  # bias for reset cell
  WH = "wh"  # weight for input -> candidate activation
  UH = "uh"  # weight for prev_state -> candidate activation
  BH = "bh"  # bias for candidate activation
  POSSIBLE_INITIALIZER_KEYS = {WZ, UZ, BZ, WR, UR, BR, WH, UH, BH}
  # Keep old name for backwards compatibility

  POSSIBLE_KEYS = POSSIBLE_INITIALIZER_KEYS

  def __init__(self, hidden_size, initializers=None, partitioners=None,
               regularizers=None, name="gru"):
    """Construct GRU.

    Args:
      hidden_size: (int) Hidden size dimensionality.
      initializers: Dict containing ops to initialize the weights. This
        dict may contain any of the keys returned by
        `GRU.get_possible_initializer_keys`.
      partitioners: Optional dict containing partitioners to partition
        the weights and biases. As a default, no partitioners are used. This
        dict may contain any of the keys returned by
        `GRU.get_possible_initializer_keys`
      regularizers: Optional dict containing regularizers for the weights and
        biases. As a default, no regularizers are used. This
        dict may contain any of the keys returned by
        `GRU.get_possible_initializer_keys`
      name: name of the module.

    Raises:
      KeyError: if `initializers` contains any keys not returned by
        `GRU.get_possible_initializer_keys`.
      KeyError: if `partitioners` contains any keys not returned by
        `GRU.get_possible_initializer_keys`.
      KeyError: if `regularizers` contains any keys not returned by
        `GRU.get_possible_initializer_keys`.
    """
    super(GRU, self).__init__(name=name)
    self._hidden_size = hidden_size
    self._initializers = util.check_initializers(
        initializers, self.POSSIBLE_INITIALIZER_KEYS)
    self._partitioners = util.check_partitioners(
        partitioners, self.POSSIBLE_INITIALIZER_KEYS)
    self._regularizers = util.check_regularizers(
        regularizers, self.POSSIBLE_INITIALIZER_KEYS)

  @classmethod
  def get_possible_initializer_keys(cls):
    """Returns the keys the dictionary of variable initializers may contain.

    The set of all possible initializer keys are:
      wz: weight for input -> update cell
      uz: weight for prev_state -> update cell
      bz: bias for update_cell
      wr: weight for input -> reset cell
      ur: weight for prev_state -> reset cell
      br: bias for reset cell
      wh: weight for input -> candidate activation
      uh: weight for prev_state -> candidate activation
      bh: bias for candidate activation

    Returns:
      Set with strings corresponding to the strings that may be passed to the
        constructor.
    """
    return super(GRU, cls).get_possible_initializer_keys(cls)

  def _build(self, inputs, prev_state):
    """Connects the GRU module into the graph.

    If this is not the first time the module has been connected to the graph,
    the Tensors provided as inputs and state must have the same final
    dimension, in order for the existing variables to be the correct size for
    their corresponding multiplications. The batch size may differ for each
    connection.

    Args:
      inputs: Tensor of size `[batch_size, input_size]`.
      prev_state: Tensor of size `[batch_size, hidden_size]`.

    Returns:
      A tuple (output, next_state) where `output` is a Tensor of size
      `[batch_size, hidden_size]` and `next_state` is a Tensor of size
      `[batch_size, hidden_size]`.

    Raises:
      ValueError: If connecting the module into the graph any time after the
        first time, and the inferred size of the inputs does not match previous
        invocations.
    """
    input_size = inputs.get_shape()[1]
    weight_shape = (input_size, self._hidden_size)
    u_shape = (self._hidden_size, self._hidden_size)
    bias_shape = (self._hidden_size,)

    self._wz = tf.get_variable(GRU.WZ, weight_shape, dtype=inputs.dtype,
                               initializer=self._initializers.get(GRU.WZ),
                               partitioner=self._partitioners.get(GRU.WZ),
                               regularizer=self._regularizers.get(GRU.WZ))
    self._uz = tf.get_variable(GRU.UZ, u_shape, dtype=inputs.dtype,
                               initializer=self._initializers.get(GRU.UZ),
                               partitioner=self._partitioners.get(GRU.UZ),
                               regularizer=self._regularizers.get(GRU.UZ))
    self._bz = tf.get_variable(GRU.BZ, bias_shape, dtype=inputs.dtype,
                               initializer=self._initializers.get(GRU.BZ),
                               partitioner=self._partitioners.get(GRU.BZ),
                               regularizer=self._regularizers.get(GRU.BZ))
    z = tf.sigmoid(tf.matmul(inputs, self._wz) +
                   tf.matmul(prev_state, self._uz) + self._bz)

    self._wr = tf.get_variable(GRU.WR, weight_shape, dtype=inputs.dtype,
                               initializer=self._initializers.get(GRU.WR),
                               partitioner=self._partitioners.get(GRU.WR),
                               regularizer=self._regularizers.get(GRU.WR))
    self._ur = tf.get_variable(GRU.UR, u_shape, dtype=inputs.dtype,
                               initializer=self._initializers.get(GRU.UR),
                               partitioner=self._partitioners.get(GRU.UR),
                               regularizer=self._regularizers.get(GRU.UR))
    self._br = tf.get_variable(GRU.BR, bias_shape, dtype=inputs.dtype,
                               initializer=self._initializers.get(GRU.BR),
                               partitioner=self._partitioners.get(GRU.BR),
                               regularizer=self._regularizers.get(GRU.BR))
    r = tf.sigmoid(tf.matmul(inputs, self._wr) +
                   tf.matmul(prev_state, self._ur) + self._br)

    self._wh = tf.get_variable(GRU.WH, weight_shape, dtype=inputs.dtype,
                               initializer=self._initializers.get(GRU.WH),
                               partitioner=self._partitioners.get(GRU.WH),
                               regularizer=self._regularizers.get(GRU.WH))
    self._uh = tf.get_variable(GRU.UH, u_shape, dtype=inputs.dtype,
                               initializer=self._initializers.get(GRU.UH),
                               partitioner=self._partitioners.get(GRU.UH),
                               regularizer=self._regularizers.get(GRU.UH))
    self._bh = tf.get_variable(GRU.BH, bias_shape, dtype=inputs.dtype,
                               initializer=self._initializers.get(GRU.BH),
                               partitioner=self._partitioners.get(GRU.BH),
                               regularizer=self._regularizers.get(GRU.BH))
    h_twiddle = tf.tanh(tf.matmul(inputs, self._wh) +
                        tf.matmul(r * prev_state, self._uh) + self._bh)

    state = (1 - z) * prev_state + z * h_twiddle
    return state, state

  @property
  def state_size(self):
    return tf.TensorShape([self._hidden_size])

  @property
  def output_size(self):
    return tf.TensorShape([self._hidden_size])
