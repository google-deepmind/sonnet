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
  out, next_state = rnn(input, rnn.initial_state(batch_size))
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


LSTMState = collections.namedtuple("LSTMState", ("hidden", "cell"))


class LSTM(rnn_core.RNNCore):
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

  #### Recurrent projections

  Projection of the recurrent state, to reduce model parameters and speed up
  computation. For more details see:

    https://arxiv.org/abs/1402.1128

  Attributes:
    state_size: Tuple of `tf.TensorShape`s indicating the size of state tensors.
    output_size: `tf.TensorShape` indicating the size of the core output.
    use_peepholes: Boolean indicating whether peephole connections are used.
  """
  # Keys that may be provided for parameter initializers.
  W_GATES = "w_gates"  # weight for gates
  B_GATES = "b_gates"  # bias of gates
  W_F_DIAG = "w_f_diag"  # weight for prev_cell -> forget gate peephole
  W_I_DIAG = "w_i_diag"  # weight for prev_cell -> input gate peephole
  W_O_DIAG = "w_o_diag"  # weight for prev_cell -> output gate peephole
  W_H_PROJECTION = "w_h_projection"  # weight for (opt) projection of h in state
  POSSIBLE_INITIALIZER_KEYS = {
      W_GATES, B_GATES, W_F_DIAG, W_I_DIAG, W_O_DIAG, W_H_PROJECTION}

  def __init__(self,
               hidden_size,
               forget_bias=1.0,
               initializers=None,
               partitioners=None,
               regularizers=None,
               use_peepholes=False,
               use_layer_norm=False,
               hidden_clip_value=None,
               projection_size=None,
               cell_clip_value=None,
               custom_getter=None,
               name="lstm"):
    """Construct LSTM.

    Args:
      hidden_size: (int) Hidden size dimensionality.
      forget_bias: (float) Bias for the forget activation.
      initializers: Dict containing ops to initialize the weights.
        This dictionary may contain any of the keys returned by
        `LSTM.get_possible_initializer_keys`.
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
      use_layer_norm: Boolean that indicates whether to apply layer
        normalization.
      hidden_clip_value: Optional number; if set, then the LSTM hidden state
        vector is clipped by this value.
      projection_size: Optional number; if set, then the LSTM hidden state is
        projected to this size via a learnable projection matrix.
      cell_clip_value: Optional number; if set, then the LSTM cell vector is
        clipped by this value.
      custom_getter: Callable that takes as a first argument the true getter,
        and allows overwriting the internal get_variable method. See the
        `tf.get_variable` documentation for more details.
      name: Name of the module.

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
    super(LSTM, self).__init__(custom_getter=custom_getter, name=name)

    self._hidden_size = hidden_size
    self._forget_bias = forget_bias
    self._use_peepholes = use_peepholes
    self._use_layer_norm = use_layer_norm
    self._hidden_clip_value = hidden_clip_value
    self._cell_clip_value = cell_clip_value
    self._use_projection = projection_size is not None
    self._hidden_state_size = projection_size or hidden_size

    self.possible_keys = self.get_possible_initializer_keys(
        use_peepholes=use_peepholes, use_projection=self._use_projection)
    self._initializers = util.check_initializers(initializers,
                                                 self.possible_keys)
    self._partitioners = util.check_initializers(partitioners,
                                                 self.possible_keys)
    self._regularizers = util.check_initializers(regularizers,
                                                 self.possible_keys)
    if hidden_clip_value is not None and hidden_clip_value < 0:
      raise ValueError("The value of hidden_clip_value should be nonnegative.")
    if cell_clip_value is not None and cell_clip_value < 0:
      raise ValueError("The value of cell_clip_value should be nonnegative.")

  @classmethod
  def get_possible_initializer_keys(cls, use_peepholes=False,
                                    use_projection=False):
    """Returns the keys the dictionary of variable initializers may contain.

    The set of all possible initializer keys are:
      w_gates:  weight for gates
      b_gates:  bias of gates
      w_f_diag: weight for prev_cell -> forget gate peephole
      w_i_diag: weight for prev_cell -> input gate peephole
      w_o_diag: weight for prev_cell -> output gate peephole

    Args:
      cls:The class.
      use_peepholes: Boolean that indicates whether peephole connections are
        used.
      use_projection: Boolean that indicates whether a recurrent projection
        layer is used.

    Returns:
      Set with strings corresponding to the strings that may be passed to the
        constructor.
    """

    possible_keys = cls.POSSIBLE_INITIALIZER_KEYS.copy()
    if not use_peepholes:
      possible_keys.difference_update(
          {cls.W_F_DIAG, cls.W_I_DIAG, cls.W_O_DIAG})
    if not use_projection:
      possible_keys.difference_update({cls.W_H_PROJECTION})
    return possible_keys

  def _build(self, inputs, prev_state):
    """Connects the LSTM module into the graph.

    If this is not the first time the module has been connected to the graph,
    the Tensors provided as inputs and state must have the same final
    dimension, in order for the existing variables to be the correct size for
    their corresponding multiplications. The batch size may differ for each
    connection.

    Args:
      inputs: Tensor of size `[batch_size, input_size]`.
      prev_state: Tuple (prev_hidden, prev_cell).

    Returns:
      A tuple (output, next_state) where 'output' is a Tensor of size
      `[batch_size, hidden_size]` and 'next_state' is a `LSTMState` namedtuple
      (next_hidden, next_cell) where `next_hidden` and `next_cell` have size
      `[batch_size, hidden_size]`. If `projection_size` is specified, then
      `next_hidden` will have size `[batch_size, projection_size]`.
    Raises:
      ValueError: If connecting the module into the graph any time after the
        first time, and the inferred size of the inputs does not match previous
        invocations.
    """
    prev_hidden, prev_cell = prev_state

    # pylint: disable=invalid-unary-operand-type
    if self._hidden_clip_value is not None:
      prev_hidden = tf.clip_by_value(
          prev_hidden, -self._hidden_clip_value, self._hidden_clip_value)
    if self._cell_clip_value is not None:
      prev_cell = tf.clip_by_value(
          prev_cell, -self._cell_clip_value, self._cell_clip_value)
    # pylint: enable=invalid-unary-operand-type

    self._create_gate_variables(inputs.get_shape(), inputs.dtype)

    # pylint false positive: calling module of same file;
    # pylint: disable=not-callable

    # Parameters of gates are concatenated into one multiply for efficiency.
    inputs_and_hidden = tf.concat([inputs, prev_hidden], 1)
    gates = tf.matmul(inputs_and_hidden, self._w_xh)

    if self._use_layer_norm:
      gates = layer_norm.LayerNorm()(gates)

    gates += self._b

    # i = input_gate, j = next_input, f = forget_gate, o = output_gate
    i, j, f, o = tf.split(value=gates, num_or_size_splits=4, axis=1)

    if self._use_peepholes:  # diagonal connections
      self._create_peephole_variables(inputs.dtype)
      f += self._w_f_diag * prev_cell
      i += self._w_i_diag * prev_cell

    forget_mask = tf.sigmoid(f + self._forget_bias)
    next_cell = forget_mask * prev_cell + tf.sigmoid(i) * tf.tanh(j)
    cell_output = next_cell
    if self._use_peepholes:
      cell_output += self._w_o_diag * cell_output
    next_hidden = tf.tanh(cell_output) * tf.sigmoid(o)

    if self._use_projection:
      next_hidden = tf.matmul(next_hidden, self._w_h_projection)

    return next_hidden, LSTMState(hidden=next_hidden, cell=next_cell)

  def _create_gate_variables(self, input_shape, dtype):
    """Initialize the variables used for the gates."""
    if len(input_shape) != 2:
      raise ValueError(
          "Rank of shape must be {} not: {}".format(2, len(input_shape)))

    equiv_input_size = self._hidden_state_size + input_shape.dims[1].value
    initializer = basic.create_linear_initializer(equiv_input_size)

    self._w_xh = tf.get_variable(
        self.W_GATES,
        shape=[equiv_input_size, 4 * self._hidden_size],
        dtype=dtype,
        initializer=self._initializers.get(self.W_GATES, initializer),
        partitioner=self._partitioners.get(self.W_GATES),
        regularizer=self._regularizers.get(self.W_GATES))
    self._b = tf.get_variable(
        self.B_GATES,
        shape=[4 * self._hidden_size],
        dtype=dtype,
        initializer=self._initializers.get(self.B_GATES, initializer),
        partitioner=self._partitioners.get(self.B_GATES),
        regularizer=self._regularizers.get(self.B_GATES))
    if self._use_projection:
      w_h_initializer = basic.create_linear_initializer(self._hidden_size)
      self._w_h_projection = tf.get_variable(
          self.W_H_PROJECTION,
          shape=[self._hidden_size, self._hidden_state_size],
          dtype=dtype,
          initializer=self._initializers.get(self.W_H_PROJECTION,
                                             w_h_initializer),
          partitioner=self._partitioners.get(self.W_H_PROJECTION),
          regularizer=self._regularizers.get(self.W_H_PROJECTION))

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

  @property
  def state_size(self):
    """Tuple of `tf.TensorShape`s indicating the size of state tensors."""
    return LSTMState(tf.TensorShape([self._hidden_state_size]),
                     tf.TensorShape([self._hidden_size]))

  @property
  def output_size(self):
    """`tf.TensorShape` indicating the size of the core output."""
    return tf.TensorShape([self._hidden_state_size])

  @property
  def use_peepholes(self):
    """Boolean indicating whether peephole connections are used."""
    return self._use_peepholes

  @property
  def use_layer_norm(self):
    """Boolean indicating whether layer norm is enabled."""
    return self._use_layer_norm


class RecurrentDropoutWrapper(rnn_core.RNNCore):
  """Wraps an RNNCore so that recurrent dropout can be applied."""

  def __init__(self, core, keep_probs):
    """Builds a new wrapper around a given core.

    Args:
      core: the RNN core to be wrapped.
      keep_probs: the recurrent dropout keep probabilities to apply.
        This should have the same structure has core.init_state. No dropout is
        applied for leafs set to None.
    """

    super(RecurrentDropoutWrapper, self).__init__(
        custom_getter=None, name=core.module_name + "_recdropout")
    self._core = core
    self._keep_probs = keep_probs

    # self._dropout_state_size is a list of shape for the state parts to which
    # dropout is to be applied.
    # self._dropout_index has the same shape as the core state. Leafs contain
    # either None if no dropout is applied or an integer representing an index
    # in self._dropout_state_size.
    self._dropout_state_size = []

    def set_dropout_state_size(keep_prob, state_size):
      if keep_prob is not None:
        self._dropout_state_size.append(state_size)
        return len(self._dropout_state_size) - 1
      return None

    self._dropout_indexes = tf.contrib.framework.nest.map_structure(
        set_dropout_state_size, keep_probs, core.state_size)

  def _build(self, inputs, prev_state):
    core_state, dropout_masks = prev_state
    output, next_core_state = self._core(inputs, core_state)

    # Dropout masks are generated via tf.nn.dropout so they actually include
    # rescaling: the mask value is 1/keep_prob if no dropout is applied.
    next_core_state = tf.contrib.framework.nest.map_structure(
        lambda i, state: state if i is None else state * dropout_masks[i],
        self._dropout_indexes, next_core_state)

    return output, (next_core_state, dropout_masks)

  def initial_state(self, batch_size, dtype=tf.float32, trainable=False,
                    trainable_initializers=None, trainable_regularizers=None,
                    name=None):
    """Builds the default start state tensor of zeros."""
    core_initial_state = self._core.initial_state(
        batch_size, dtype=dtype, trainable=trainable,
        trainable_initializers=trainable_initializers,
        trainable_regularizers=trainable_regularizers, name=name)

    dropout_masks = [None] * len(self._dropout_state_size)
    def set_dropout_mask(index, state, keep_prob):
      if index is not None:
        ones = tf.ones_like(state, dtype=dtype)
        dropout_masks[index] = tf.nn.dropout(ones, keep_prob=keep_prob)
    tf.contrib.framework.nest.map_structure(
        set_dropout_mask,
        self._dropout_indexes, core_initial_state, self._keep_probs)

    return core_initial_state, dropout_masks

  @property
  def state_size(self):
    return self._core.state_size, self._dropout_state_size

  @property
  def output_size(self):
    return self._core.output_size


def lstm_with_recurrent_dropout(hidden_size, keep_prob=0.5, **kwargs):
  """LSTM with recurrent dropout.

  Args:
    hidden_size: the LSTM hidden size.
    keep_prob: the probability to keep an entry when applying dropout.
    **kwargs: Extra keyword arguments to pass to the LSTM.

  Returns:
    A tuple (train_lstm, test_lstm) where train_lstm is an LSTM with
    recurrent dropout enabled to be used for training and test_lstm
    is the same LSTM without recurrent dropout.
  """

  lstm = LSTM(hidden_size, **kwargs)
  return RecurrentDropoutWrapper(lstm, LSTMState(keep_prob, None)), lstm


class ZoneoutWrapper(rnn_core.RNNCore):
  """Wraps an RNNCore so that zoneout can be applied.

  Zoneout was introduced in https://arxiv.org/abs/1606.01305
  It consists of randomly freezing some RNN state in the same way recurrent
  dropout would replace this state with zero.
  """

  def __init__(self, core, keep_probs, is_training):
    """Builds a new wrapper around a given core.

    Args:
      core: the RNN core to be wrapped.
      keep_probs: the probabilities to use the updated states rather than
        keeping the old state values. This is one minus the probability
        that zoneout gets applied.
        This should have the same structure has core.init_state. No zoneout is
        applied for leafs set to None.
      is_training: when set, apply some stochastic zoneout. Otherwise perform
        a linear combination of the previous state and the current state based
        on the zoneout probability.
    """

    super(ZoneoutWrapper, self).__init__(
        custom_getter=None, name=core.module_name + "_zoneout")
    self._core = core
    self._keep_probs = keep_probs
    self._is_training = is_training

  def _build(self, inputs, prev_state):
    output, next_state = self._core(inputs, prev_state)

    def apply_zoneout(keep_prob, next_s, prev_s):  # pylint: disable=missing-docstring
      if keep_prob is None:
        return next_s
      if self._is_training:
        diff = next_s - prev_s
        # The dropout returns 0 with probability 1 - keep_prob and in this case
        # this function returns prev_s
        # It returns diff / keep_prob otherwise and then this function returns
        # prev_s + diff = next_s
        return prev_s + tf.nn.dropout(diff, keep_prob) * keep_prob
      else:
        return prev_s * (1 - keep_prob) + next_s * keep_prob

    next_state = tf.contrib.framework.nest.map_structure(
        apply_zoneout, self._keep_probs, next_state, prev_state)

    return output, next_state

  def initial_state(self, batch_size, dtype=tf.float32, trainable=False,
                    trainable_initializers=None, trainable_regularizers=None,
                    name=None):
    """Builds the default start state tensor of zeros."""
    return self._core.initial_state(
        batch_size, dtype=dtype, trainable=trainable,
        trainable_initializers=trainable_initializers,
        trainable_regularizers=trainable_regularizers, name=name)

  @property
  def state_size(self):
    return self._core.state_size

  @property
  def output_size(self):
    return self._core.output_size


def lstm_with_zoneout(hidden_size, keep_prob_c=0.5, keep_prob_h=0.95, **kwargs):
  """LSTM with recurrent dropout.

  Args:
    hidden_size: the LSTM hidden size.
    keep_prob_c: the probability to use the new value of the cell state rather
      than freezing it.
    keep_prob_h: the probability to use the new value of the hidden state
      rather than freezing it.
    **kwargs: Extra keyword arguments to pass to the LSTM.

  Returns:
    A tuple (train_lstm, test_lstm) where train_lstm is an LSTM with
    recurrent dropout enabled to be used for training and test_lstm
    is the same LSTM without zoneout.
  """

  lstm = LSTM(hidden_size, **kwargs)
  keep_probs = LSTMState(keep_prob_h, keep_prob_c)
  train_lstm = ZoneoutWrapper(lstm, keep_probs, is_training=True)
  test_lstm = ZoneoutWrapper(lstm, keep_probs, is_training=False)
  return train_lstm, test_lstm


class BatchNormLSTM(rnn_core.RNNCore):
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
               custom_getter=None,
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
      custom_getter: Callable that takes as a first argument the true getter,
        and allows overwriting the internal get_variable method. See the
        `tf.get_variable` documentation for more details.
      name: Name of the module.

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
    super(BatchNormLSTM, self).__init__(custom_getter=custom_getter, name=name)

    self._hidden_size = hidden_size
    self._forget_bias = forget_bias
    self._use_peepholes = use_peepholes
    self._max_unique_stats = max_unique_stats
    self._use_batch_norm_h = use_batch_norm_h
    self._use_batch_norm_x = use_batch_norm_x
    self._use_batch_norm_c = use_batch_norm_c
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
    if hidden_clip_value is not None and hidden_clip_value < 0:
      raise ValueError("The value of hidden_clip_value should be nonnegative.")
    if cell_clip_value is not None and cell_clip_value < 0:
      raise ValueError("The value of cell_clip_value should be nonnegative.")

    if use_batch_norm_h:
      self._batch_norm_h = BatchNormLSTM.IndexedStatsBatchNorm(max_unique_stats,
                                                               "batch_norm_h")
    if use_batch_norm_x:
      self._batch_norm_x = BatchNormLSTM.IndexedStatsBatchNorm(max_unique_stats,
                                                               "batch_norm_x")
    if use_batch_norm_c:
      self._batch_norm_c = BatchNormLSTM.IndexedStatsBatchNorm(max_unique_stats,
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
      snt.RNNCore wrapping this class with the extra input(s) added.
    """
    return BatchNormLSTM.CoreWithExtraBuildArgs(
        self, is_training=is_training, test_local_stats=test_local_stats)

  @classmethod
  def get_possible_initializer_keys(
      cls, use_peepholes=False, use_batch_norm_h=True, use_batch_norm_x=False,
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
    if is_training is None:
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

    gates += self._b

    # i = input_gate, j = next_input, f = forget_gate, o = output_gate
    i, j, f, o = tf.split(value=gates, num_or_size_splits=4, axis=1)

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
      return super(BatchNormLSTM, self).initial_state(
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
      super(BatchNormLSTM.IndexedStatsBatchNorm, self).__init__(name=name)
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

  class CoreWithExtraBuildArgs(rnn_core.RNNCore):
    """Wraps an RNNCore so that the build method receives extra args and kwargs.

    This will pass the additional input `args` and `kwargs` to the _build
    function of the snt.RNNCore after the input and prev_state inputs.
    """

    def __init__(self, core, *args, **kwargs):
      """Construct the CoreWithExtraBuildArgs.

      Args:
        core: The snt.RNNCore to wrap.
        *args: Extra arguments to pass to _build.
        **kwargs: Extra keyword arguments to pass to _build.
      """
      super(BatchNormLSTM.CoreWithExtraBuildArgs, self).__init__(
          name=core.module_name + "_extra_args")
      self._core = core
      self._args = args
      self._kwargs = kwargs

    def _build(self, inputs, state):
      return self._core(inputs, state, *self._args, **self._kwargs)

    @property
    def state_size(self):
      """Tuple indicating the size of nested state tensors."""
      return self._core.state_size

    @property
    def output_size(self):
      """`tf.TensorShape` indicating the size of the core output."""
      return self._core.output_size


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
               rate=1,
               padding=conv.SAME,
               use_bias=True,
               legacy_bias_behaviour=True,
               forget_bias=1.0,
               initializers=None,
               partitioners=None,
               regularizers=None,
               use_layer_norm=False,
               custom_getter=None,
               name="conv_lstm"):
    """Construct ConvLSTM.

    Args:
      conv_ndims: Convolution dimensionality (1, 2 or 3).
      input_shape: Shape of the input as an iterable, excluding the batch size.
      output_channels: Number of output channels of the conv LSTM.
      kernel_shape: Sequence of kernel sizes (of size conv_ndims), or integer
          that is used to define kernel size in all dimensions.
      stride: Sequence of kernel strides (of size conv_ndims), or integer that
          is used to define stride in all dimensions.
      rate: Sequence of dilation rates (of size conv_ndims), or integer that is
          used to define dilation rate in all dimensions. 1 corresponds to a
          standard convolution, while rate > 1 corresponds to a dilated
          convolution. Cannot be > 1 if any of stride is also > 1.
      padding: Padding algorithm, either `snt.SAME` or `snt.VALID`.
      use_bias: Use bias in convolutions.
      legacy_bias_behaviour: If True, bias is applied to both input and hidden
        convolutions, creating a redundant bias variable. If False, bias is only
        applied to input convolution, removing the redundancy.
      forget_bias: Forget bias.
      initializers: Dict containing ops to initialize the convolutional weights.
      partitioners: Optional dict containing partitioners to partition
        the convolutional weights and biases. As a default, no partitioners are
        used.
      regularizers: Optional dict containing regularizers for the convolutional
        weights and biases. As a default, no regularizers are used.
      use_layer_norm: Boolean that indicates whether to apply layer
        normalization. This is applied across the entire layer, normalizing
        over all non-batch dimensions.
      custom_getter: Callable that takes as a first argument the true getter,
        and allows overwriting the internal get_variable method. See the
        `tf.get_variable` documentation for more details.
      name: Name of the module.

    Raises:
      ValueError: If `skip_connection` is `True` and stride is different from 1
        or if `input_shape` is incompatible with `conv_ndims`.
    """
    super(ConvLSTM, self).__init__(custom_getter=custom_getter, name=name)

    self._conv_class = self._get_conv_class(conv_ndims)

    if conv_ndims != len(input_shape)-1:
      raise ValueError("Invalid input_shape {} for conv_ndims={}.".format(
          input_shape, conv_ndims))

    self._conv_ndims = conv_ndims
    self._input_shape = tuple(input_shape)
    self._output_channels = output_channels
    self._kernel_shape = kernel_shape
    self._stride = stride
    self._rate = rate
    self._padding = padding
    self._use_bias = use_bias
    self._legacy_bias_behaviour = legacy_bias_behaviour
    self._forget_bias = forget_bias
    self._initializers = initializers
    self._partitioners = partitioners
    self._regularizers = regularizers
    if use_layer_norm:
      util.deprecation_warning(
          "`use_layer_norm` kwarg is being deprecated as the implementation is "
          "currently incorrect - scale and offset params are created for "
          "spatial_dims * channels instead of just channels.")
    self._use_layer_norm = use_layer_norm

    self._total_output_channels = output_channels
    if self._stride != 1:
      self._total_output_channels //= self._stride * self._stride

    self._convolutions = dict()


    if self._use_bias and self._legacy_bias_behaviour:
      tf.logging.warning(
          "ConvLSTM will create redundant bias variables for input and hidden "
          "convolutions. To avoid this, invoke the constructor with option "
          "`legacy_bias_behaviour=False`. In future, this will be the default.")

  def _new_convolution(self, use_bias):
    """Returns new convolution.

    Args:
      use_bias: Use bias in convolutions. If False, clean_dict removes bias
        entries from initializers, partitioners and regularizers passed to
        the constructor of the convolution.
    """
    def clean_dict(input_dict):
      if input_dict and not use_bias:
        cleaned_dict = input_dict.copy()
        cleaned_dict.pop("b", None)
        return cleaned_dict
      return input_dict
    return self._conv_class(
        output_channels=4*self._output_channels,
        kernel_shape=self._kernel_shape,
        stride=self._stride,
        rate=self._rate,
        padding=self._padding,
        use_bias=use_bias,
        initializers=clean_dict(self._initializers),
        partitioners=clean_dict(self._partitioners),
        regularizers=clean_dict(self._regularizers),
        name="conv")

  @property
  def convolutions(self):
    return self._convolutions

  @property
  def state_size(self):
    """Tuple of `tf.TensorShape`s indicating the size of state tensors."""
    hidden_size = tf.TensorShape(
        self._input_shape[:-1] + (self._output_channels,))
    return (hidden_size, hidden_size)

  @property
  def output_size(self):
    """`tf.TensorShape` indicating the size of the core output."""
    return tf.TensorShape(
        self._input_shape[:-1] + (self._total_output_channels,))

  def _build(self, inputs, state):
    hidden, cell = state
    if "input" not in self._convolutions:
      self._convolutions["input"] = self._new_convolution(self._use_bias)
    if "hidden" not in self._convolutions:
      if self._legacy_bias_behaviour:
        self._convolutions["hidden"] = self._new_convolution(self._use_bias)
      else:
        # Do not apply bias a second time
        self._convolutions["hidden"] = self._new_convolution(use_bias=False)
    input_conv = self._convolutions["input"]
    hidden_conv = self._convolutions["hidden"]
    next_hidden = input_conv(inputs) + hidden_conv(hidden)

    if self._use_layer_norm:
      # Normalize over all non-batch dimensions.
      # Temporarily flatten the spatial and channel dimensions together.
      flatten = basic.BatchFlatten()
      unflatten = basic.BatchReshape(next_hidden.get_shape().as_list()[1:])
      next_hidden = flatten(next_hidden)
      next_hidden = layer_norm.LayerNorm()(next_hidden)
      next_hidden = unflatten(next_hidden)

    gates = tf.split(value=next_hidden, num_or_size_splits=4,
                     axis=self._conv_ndims+1)

    input_gate, next_input, forget_gate, output_gate = gates
    next_cell = tf.sigmoid(forget_gate + self._forget_bias) * cell
    next_cell += tf.sigmoid(input_gate) * tf.tanh(next_input)
    output = tf.tanh(next_cell) * tf.sigmoid(output_gate)
    return output, (output, next_cell)

  @property
  def use_layer_norm(self):
    """Boolean indicating whether layer norm is enabled."""
    return self._use_layer_norm


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

  def __init__(self, hidden_size, initializers=None, partitioners=None,
               regularizers=None, custom_getter=None, name="gru"):
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
      custom_getter: Callable that takes as a first argument the true getter,
        and allows overwriting the internal get_variable method. See the
        `tf.get_variable` documentation for more details.
      name: Name of the module.

    Raises:
      KeyError: if `initializers` contains any keys not returned by
        `GRU.get_possible_initializer_keys`.
      KeyError: if `partitioners` contains any keys not returned by
        `GRU.get_possible_initializer_keys`.
      KeyError: if `regularizers` contains any keys not returned by
        `GRU.get_possible_initializer_keys`.
    """
    super(GRU, self).__init__(custom_getter=custom_getter, name=name)
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


class HighwayCore(rnn_core.RNNCore):
  """Recurrent Highway Network cell.

  The implementation is based on: https://arxiv.org/pdf/1607.03474v5.pdf
  As per the first lines of section 5 of the reference paper, 1 - T is
  used instead of a dedicated C gate.

  Attributes:
    state_size: Integer indicating the size of state tensor.
    output_size: Integer indicating the size of the core output.
  """

  # Keys that may be provided for parameter initializers.
  WT = "wt"  # weight for input or previous state -> T gate
  BT = "bt"  # bias for previous state -> T gate
  WH = "wh"  # weight for input or previous state -> H gate
  BH = "bh"  # bias for previous state -> H gate

  def __init__(
      self,
      hidden_size,
      num_layers,
      initializers=None,
      partitioners=None,
      regularizers=None,
      custom_getter=None,
      name="highwaycore"):
    """Construct a new Recurrent Highway core.

    Args:
      hidden_size: (int) Hidden size dimensionality.
      num_layers: (int) Number of highway layers.
      initializers: Dict containing ops to initialize the weights. This
        dict may contain any of the keys returned by
        `HighwayCore.get_possible_initializer_keys`.
      partitioners: Optional dict containing partitioners to partition
        the weights and biases. As a default, no partitioners are used. This
        dict may contain any of the keys returned by
        `HighwayCore.get_possible_initializer_keys`.
      regularizers: Optional dict containing regularizers for the weights and
        biases. As a default, no regularizers are used. This
        dict may contain any of the keys returned by
        `HighwayCore.get_possible_initializer_keys`.
      custom_getter: Callable that takes as a first argument the true getter,
        and allows overwriting the internal get_variable method. See the
        `tf.get_variable` documentation for more details.
      name: Name of the module.

    Raises:
      KeyError: if `initializers` contains any keys not returned by
        `HighwayCore.get_possible_initializer_keys`.
      KeyError: if `partitioners` contains any keys not returned by
        `HighwayCore.get_possible_initializer_keys`.
      KeyError: if `regularizers` contains any keys not returned by
        `HighwayCore.get_possible_initializer_keys`.
    """
    super(HighwayCore, self).__init__(custom_getter=custom_getter, name=name)
    self._hidden_size = hidden_size
    self._num_layers = num_layers
    self._initializers = util.check_initializers(
        initializers, self.get_possible_initializer_keys(num_layers))
    self._partitioners = util.check_partitioners(
        partitioners, self.get_possible_initializer_keys(num_layers))
    self._regularizers = util.check_regularizers(
        regularizers, self.get_possible_initializer_keys(num_layers))

  @classmethod
  def get_possible_initializer_keys(cls, num_layers):
    """Returns the keys the dictionary of variable initializers may contain.

    The set of all possible initializer keys are:
      wt: weight for input -> T gate
      wh: weight for input -> H gate
      wtL: weight for prev state -> T gate for layer L (indexed from 0)
      whL: weight for prev state -> H gate for layer L (indexed from 0)
      btL: bias for prev state -> T gate for layer L (indexed from 0)
      bhL: bias for prev state -> H gate for layer L (indexed from 0)

    Args:
      num_layers: (int) Number of highway layers.
    Returns:
      Set with strings corresponding to the strings that may be passed to the
        constructor.
    """
    keys = [cls.WT, cls.WH]
    for layer_index in xrange(num_layers):
      layer_str = str(layer_index)
      keys += [
          cls.WT + layer_str,
          cls.BT + layer_str,
          cls.WH + layer_str,
          cls.BH + layer_str]
    return set(keys)

  def _build(self, inputs, prev_state):
    """Connects the highway core module into the graph.

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

    def _get_variable(name, shape):
      return tf.get_variable(
          name,
          shape,
          dtype=inputs.dtype,
          initializer=self._initializers.get(name),
          partitioner=self._partitioners.get(name),
          regularizer=self._regularizers.get(name))

    pre_highway_wt = _get_variable(self.WT, weight_shape)
    pre_highway_wh = _get_variable(self.WH, weight_shape)
    state = prev_state
    for layer_index in xrange(self._num_layers):
      layer_str = str(layer_index)
      layer_wt = _get_variable(self.WT + layer_str, u_shape)
      layer_bt = _get_variable(self.BT + layer_str, bias_shape)
      layer_wh = _get_variable(self.WH + layer_str, u_shape)
      layer_bh = _get_variable(self.BH + layer_str, bias_shape)
      linear_t = tf.matmul(state, layer_wt) + layer_bt
      linear_h = tf.matmul(state, layer_wh) + layer_bh
      if layer_index == 0:
        linear_t += tf.matmul(inputs, pre_highway_wt)
        linear_h += tf.matmul(inputs, pre_highway_wh)
      output_t = tf.sigmoid(linear_t)
      output_h = tf.tanh(linear_h)
      state = state * (1 - output_t) + output_h * output_t

    return state, state

  @property
  def state_size(self):
    return tf.TensorShape([self._hidden_size])

  @property
  def output_size(self):
    return tf.TensorShape([self._hidden_size])


def highway_core_with_recurrent_dropout(
    hidden_size,
    num_layers,
    keep_prob=0.5,
    **kwargs):
  """Highway core with recurrent dropout.

  Args:
    hidden_size: (int) Hidden size dimensionality.
    num_layers: (int) Number of highway layers.
    keep_prob: the probability to keep an entry when applying dropout.
    **kwargs: Extra keyword arguments to pass to the highway core.

  Returns:
    A tuple (train_core, test_core) where train_core is a higway core with
    recurrent dropout enabled to be used for training and test_core is the
    same highway core without recurrent dropout.
  """

  core = HighwayCore(hidden_size, num_layers, **kwargs)
  return RecurrentDropoutWrapper(core, keep_prob), core


class LSTMBlockCell(rnn_core.RNNCellWrapper):
  """Wraps the TensorFlow LSTMBlockCell as a Sonnet RNNCore."""

  @rnn_core.with_doc(tf.contrib.rnn.LSTMBlockCell.__init__)
  def __init__(self, *args, **kwargs):
    super(LSTMBlockCell, self).__init__(tf.contrib.rnn.LSTMBlockCell,
                                        *args, **kwargs)
