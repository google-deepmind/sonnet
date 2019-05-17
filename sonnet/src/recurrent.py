# Copyright 2019 The Sonnet Authors. All Rights Reserved.
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

"""Recurrent Neural Network cores."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import functools

import six
from sonnet.src import base
from sonnet.src import conv
from sonnet.src import initializers
from sonnet.src import linear
from sonnet.src import once
from sonnet.src import utils
import tensorflow.compat.v1 as tf1
import tensorflow as tf

# A temporary import until tree is open-sourced.
# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.util import nest


@six.add_metaclass(abc.ABCMeta)
class RNNCore(base.Module):
  """Base class for Recurrent Neural Network cores.

  This class defines the basic functionality that every core should
  implement: `initial_state` ,used to construct an example of the core
  state; and `__call__` which applies the core parameterized by a
  previous state to an input.

  Cores are typically used with `snt.*_unroll` to iteratively construct
  an output sequence from the given input sequence.
  """

  @abc.abstractmethod
  def __call__(self, inputs, prev_state):
    """Perform one step of an RNN.

    Args:
      inputs: An arbitrarily nested structure of shape `[B, ...]` where B
        is the batch size.
      prev_state: Previous core state.

    Returns:
      outputs: An arbitrarily nested structure of shape `[B, ...]`.
        Dimensions following the batch size could be different from that
        of `inputs`.
      next_state: next core state, must be of the same shape as the
        previous one.
    """

  @abc.abstractmethod
  def initial_state(self, batch_size, **kwargs):
    """Construct an initial state for this core.

    Args:
      batch_size: An int or an integral scalar tensor representing
        batch size.
      **kwargs: Optional keyword arguments.

    Returns:
      Arbitrarily nested initial state for this core.
    """


class TrainableState(base.Module):
  """Trainable state for an `RNNCore`.

  The state can be constructed manually from a nest of initial values

      snt.TrainableState((tf.zeros([16]), tf.zeros([16])))

  or automatically for a given `RNNCore`

      core = snt.LSTM(hidden_size=16)
      snt.TrainableState.for_core(core)
  """

  @classmethod
  def for_core(cls, core, mask=None, name=None):
    """Construct a trainable state for a given `RNNCore`.

    Args:
      core: `RNNCore` to construct the state for.
      mask: Optional boolean mask of the same structure as the initial
        state of `core` specifying which components should be trainable.
        If not given, the whole state is considered trainable.
      name: Name of the module.

    Returns:
      A `TrainableState`.
    """
    initial_values = nest.map_structure(
        lambda s: tf.squeeze(s, axis=0),
        core.initial_state(batch_size=1))
    return cls(initial_values, mask, name)

  def __init__(self, initial_values, mask=None, name=None):
    """Construct a trainable state from initial values.

    Args:
      initial_values: Arbitrarily nested initial values for the state.
      mask: Optional boolean mask of the same structure as `initial_values`
        specifying which components should be trainable. If not given,
        the whole state is considered trainable.
      name: Name of the module.
    """
    super(TrainableState, self).__init__(name)

    flat_initial_values = nest.flatten_with_joined_string_paths(initial_values)
    if mask is None:
      flat_mask = [True] * len(flat_initial_values)
    else:
      nest.assert_same_structure(initial_values, mask)
      flat_mask = nest.flatten(mask)

    flat_template = []
    for (path, initial_value), trainable in zip(flat_initial_values, flat_mask):
      # `"state"` is only used if initial_values is not nested.
      name = path or "state"
      flat_template.append(tf.Variable(
          tf.expand_dims(initial_value, axis=0),
          trainable=trainable,
          name=name))

    self._template = nest.pack_sequence_as(initial_values, flat_template)

  def __call__(self, batch_size):
    """Return a trainable state for the given batch size."""
    return nest.map_structure(
        lambda s: tf.tile(s, [batch_size] + [1]*(s.shape.rank - 1)),
        self._template)


def static_unroll(
    core,
    input_sequence,  # time-major.
    initial_state,
    sequence_length=None):
  """Perform a static unroll of an RNN.

  >>> core = snt.LSTM(hidden_size=16)
  >>> batch_size = 3
  >>> input_sequence = tf.random.uniform([1, batch_size, 2])
  >>> output_sequence, final_state = snt.static_unroll(
  ...     core,
  ...     input_sequence,
  ...     core.initial_state(batch_size))

  An *unroll* corresponds to calling an RNN on each element of the
  input sequence in a loop, carrying the state through:

      state = initial_state
      for t in range(len(input_sequence)):
         outputs, state = core(input_sequence[t], state)

  A *static* unroll replaces a loop with its body repeated multiple
  times when executed inside `tf.function`:

      state = initial_state
      outputs0, state = core(input_sequence[0], state)
      outputs1, state = core(input_sequence[1], state)
      outputs2, state = core(input_sequence[2], state)
      ...

  See `snt.dynamic_unroll` for a loop-preserving unroll function.

  Args:
    core: An `RNNCore` to unroll.
    input_sequence: An arbitrarily nested structure of tensors of shape
      `[T, B, ...]` where T is the number of time steps, and B is the
      batch size.
    initial_state: initial state of the given core.
    sequence_length: An optional tensor of shape `[B]` specifying the
      lengths of sequences within the (padded) batch.

  Returns:
    output_sequence: An arbitrarily nested structure of tensors of shape
      `[T, B, ...]`. Dimensions following the batch size could be
      different from that of `input`.
    final_state: Core state at time step T.

  Raises:
    ValueError: if `input_sequence` is empty.
  """
  num_steps, input_tas = _unstack_input_sequence(input_sequence)

  outputs = None
  state = initial_state
  output_accs = None
  for t in six.moves.range(num_steps):
    outputs, state = _rnn_step(
        core,
        input_tas,
        sequence_length,
        t,
        prev_outputs=outputs,
        prev_state=state)
    if t == 0:
      output_accs = nest.map_structure(lambda o: _ListWrapper([o]), outputs)
    else:
      nest.map_structure(
          lambda acc, o: acc.data.append(o),
          output_accs,
          outputs)

  output_sequence = nest.map_structure(
      lambda acc: tf.stack(acc.data),
      output_accs)
  return output_sequence, state


class _ListWrapper(object):
  """A wrapper hiding a list from `nest`.

  This allows to use `nest.map_structure` without recursing into the
  wrapped list.
  """

  __slots__ = ["data"]

  def __init__(self, data):
    self.data = data


# TODO(slebedev): core can be core_fn: Callable[[I, S], Tuple[O, S]].
# TODO(slebedev): explain sequence_length with ASCII art?
def dynamic_unroll(
    core,
    input_sequence,  # time-major.
    initial_state,
    sequence_length=None,
    parallel_iterations=1,
    swap_memory=False):
  """Perform a dynamic unroll of an RNN.

  >>> core = snt.LSTM(hidden_size=16)
  >>> batch_size = 3
  >>> input_sequence = tf.random.uniform([1, batch_size, 2])
  >>> output_sequence, final_state = snt.dynamic_unroll(
  ...     core,
  ...     input_sequence,
  ...     core.initial_state(batch_size))

  An *unroll* corresponds to calling an RNN on each element of the
  input sequence in a loop, carrying the state through:

      state = initial_state
      for t in range(len(input_sequence)):
         outputs, state = core(input_sequence[t], state)

  A *dynamic* unroll preserves the loop structure when executed within
  `tf.function`. See `snt.static_unroll` for an unroll function which
  replaces a loop with its body repeated multiple times.

  Args:
    core: An `RNNCore` to unroll.
    input_sequence: An arbitrarily nested structure of tensors of shape
      `[T, B, ...]` where T is the number of time steps, and B is the
      batch size.
    initial_state: initial state of the given core.
    sequence_length: An optional tensor of shape `[B]` specifying the
      lengths of sequences within the (padded) batch.
    parallel_iterations: An optional int specifying the number of
      iterations to run in parallel. Those operations which do not have
      any temporal dependency and can be run in parallel, will be. This
      parameter trades off time for space. Values >> 1 use more memory
      but take less time, while smaller values use less memory but
      computations take longer. Defaults to 1.
    swap_memory: Transparently swap the tensors produced in forward
      inference but needed for back prop from GPU to CPU. This allows
      training RNNs which would typically not fit on a single GPU,
      with very minimal (or no) performance penalty. Defaults to False.

  Returns:
    output_sequence: An arbitrarily nested structure of tensors of shape
      `[T, B, ...]`. Dimensions following the batch size could be
      different from that of `input`.
    final_state: Core state at time step T.

  Raises:
    ValueError: if `input_sequence` is empty.
  """
  num_steps, input_tas = _unstack_input_sequence(input_sequence)

  # Unroll the first time step separately to infer outputs structure.
  outputs, state = _rnn_step(
      core,
      input_tas,
      sequence_length,
      t=0,
      prev_outputs=None,
      prev_state=initial_state)
  output_tas = nest.map_structure(
      lambda o: tf.TensorArray(o.dtype, num_steps).write(0, o),
      outputs)

  # AutoGraph converts a for loop over `tf.range` to `tf.while_loop`.
  # `maximum_iterations` are needed to backprop through the loop on TPU.
  for t in tf.range(1, num_steps):
    tf.autograph.experimental.set_loop_options(
        parallel_iterations=parallel_iterations,
        swap_memory=swap_memory,
        maximum_iterations=num_steps - 1)
    outputs, state = _rnn_step(
        core,
        input_tas,
        sequence_length,
        t,
        prev_outputs=outputs,
        prev_state=state)
    output_tas = nest.map_structure(
        lambda ta, o, _t=t: ta.write(_t, o),
        output_tas,
        outputs)

  output_sequence = nest.map_structure(tf.TensorArray.stack, output_tas)
  return output_sequence, state


def _unstack_input_sequence(input_sequence):
  """Unstack the input sequence into a nest of `tf.TensorArray`s.

  This allows to traverse the input sequence using `tf.TensorArray.read`
  instead of a slice, avoiding O(sliced tensor) slice gradient
  computation during the backwards pass.

  Args:
    input_sequence: See `dynamic_unroll` or `static_unroll`.

  Returns:
    num_steps: Number of steps in the input sequence.
    input_tas: An arbitrarily nested structure of tf.TensorArrays of
      size `num_steps`.

  Raises:
    ValueError: If tensors in `input_sequence` have inconsistent number
      of steps or the number of steps is 0.
  """
  all_num_steps = {i.shape[0] for i in nest.flatten(input_sequence)}
  if len(all_num_steps) > 1:
    raise ValueError(
        "input_sequence tensors must have consistent number of time steps")
  [num_steps] = all_num_steps
  if num_steps == 0:
    raise ValueError("input_sequence must have at least a single time step")

  input_tas = nest.map_structure(
      lambda i: tf.TensorArray(i.dtype, num_steps).unstack(i),
      input_sequence)
  return num_steps, input_tas


def _safe_where(condition, x, y):  # pylint: disable=g-doc-args
  """`tf.where` which allows scalar inputs."""
  if x.shape.rank == 0:
    # This is to match the `tf.nn.*_rnn` behavior. In general, we might
    # want to branch on `tf.reduce_all(condition)`.
    return y
  # TODO(tomhennigan) Broadcasting with SelectV2 is currently broken.
  return tf1.where(condition, x, y)


def _rnn_step(
    core,
    input_tas,
    sequence_length,
    t,
    prev_outputs,
    prev_state):
  """Perform a single step of an RNN optionally accounting for variable length."""
  outputs, state = core(
      nest.map_structure(lambda i: i.read(t), input_tas),
      prev_state)

  if prev_outputs is None:
    assert t == 0
    prev_outputs = nest.map_structure(tf.zeros_like, outputs)

  # TODO(slebedev): do not go into this block if t < min_len.
  if sequence_length is not None:
    # Selectively propagate outputs/state to the not-yet-finished
    # sequences.
    maybe_propagate = functools.partial(_safe_where, sequence_length <= t)
    outputs = nest.map_structure(
        maybe_propagate,
        prev_outputs,
        outputs)
    state = nest.map_structure(maybe_propagate, prev_state, state)

  return outputs, state


class VanillaRNN(RNNCore):
  """Basic fully-connected RNN core.

  Given x_t and the previous hidden state h_{t-1} the core computes

    h_t = w_i x_t + w_h h_{t-1} + b

  Variables:
    input_to_hidden/w: weights w_i, a Tensor of shape
      [input_size, hidden_size].
    hidden_to_hidden/w: weights w_h, a Tensor of shape
      [input_size, hidden_size].
    b: bias, a Tensor or shape [hidden_size].
  """

  def __init__(
      self,
      hidden_size,
      activation=tf.tanh,
      w_i_init=None,
      w_h_init=None,
      b_init=None,
      dtype=tf.float32,
      name=None):
    """Construct a vanilla RNN core.

    Args:
      hidden_size: Hidden layer size.
      activation: Activation function to use. Defaults to `tf.tanh`.
      w_i_init: Optional initializer for the input-to-hidden weights.
        Defaults to `TruncatedNormal` with a standard deviation of
        `1 / sqrt(input_size).
      w_h_init: Optional initializer for the hidden-to-hidden weights.
        Defaults to `TruncatedNormal` with a standard deviation of
        `1 / sqrt(hidden_size).
      b_init: Optional initializer for the bias. Defaults to `Zeros`.
      dtype: Optional `tf.DType` of the core's variables. Defaults to
        `tf.float32`.
      name: Name of the module.
    """
    super(VanillaRNN, self).__init__(name)
    self._hidden_size = hidden_size
    self._activation = activation
    self._b_init = b_init or initializers.Zeros()
    self._dtype = dtype

    self._input_to_hidden = linear.Linear(
        hidden_size,
        with_bias=False,
        w_init=w_i_init,
        name="input_to_hidden")
    self._hidden_to_hidden = linear.Linear(
        hidden_size,
        with_bias=False,
        w_init=w_h_init,
        name="hidden_to_hidden")

  @property
  def input_to_hidden(self):
    return self._input_to_hidden.w

  @property
  def hidden_to_hidden(self):
    return self._hidden_to_hidden.w

  def __call__(self, inputs, prev_state):
    """See base class."""
    self._create_parameters(inputs)

    outputs = self._activation(
        self._input_to_hidden(inputs)
        + self._hidden_to_hidden(prev_state)
        + self._b)

    # For VanillaRNN, the next state of the RNN is the same as the outputs.
    return outputs, outputs

  def initial_state(self, batch_size):
    """See base class."""
    return tf.zeros(shape=[batch_size, self._hidden_size], dtype=self._dtype)

  @once.once
  def _create_parameters(self, inputs):
    dtype = _check_inputs_dtype(inputs, self._dtype)
    self._b = tf.Variable(self._b_init([self._hidden_size], dtype), name="b")


class _LegacyDeepRNN(RNNCore):
  """Sonnet 1 compatible `DeepRNN` implementation.

  This class is not intended to be used directly. Refer to `DeepRNN`
  and `deep_rnn_with_*_connections`.
  """

  def __init__(
      self,
      layers,
      skip_connections,
      concat_final_output_if_skip=True,
      name=None):
    """Construct a DeepRNN.

    Args:
      layers: A list of `RNNCore`s or callables.
      skip_connections: See `deep_rnn_with_skip_connections`.
      concat_final_output_if_skip: See `deep_rnn_with_skip_connections`.
      name: Name of the module.
    """
    super(_LegacyDeepRNN, self).__init__(name)
    self._layers = layers if layers is not None else []
    self._skip_connections = skip_connections
    self._concat_final_output_if_skip = concat_final_output_if_skip

  def __call__(self, inputs, prev_state):
    """See base class."""
    current_inputs = inputs
    outputs = []
    next_states = []
    recurrent_idx = 0
    concat = lambda *args: tf.concat(args, axis=-1)
    for idx, layer in enumerate(self._layers):
      if self._skip_connections and idx > 0:
        current_inputs = nest.map_structure(concat, inputs, current_inputs)

      if isinstance(layer, RNNCore):
        current_inputs, next_state = layer(
            current_inputs,
            prev_state[recurrent_idx])
        next_states.append(next_state)
        recurrent_idx += 1
      else:
        current_inputs = layer(current_inputs)

      if self._skip_connections:
        outputs.append(current_inputs)

    if self._skip_connections and self._concat_final_output_if_skip:
      outputs = nest.map_structure(concat, *outputs)
    else:
      outputs = current_inputs

    return outputs, tuple(next_states)

  def initial_state(self, batch_size, **kwargs):
    """See base class."""
    return tuple(
        layer.initial_state(batch_size, **kwargs)
        for layer in self._layers
        if isinstance(layer, RNNCore))


class DeepRNN(_LegacyDeepRNN):
  """Linear chain of modules or callables.

  The core takes `(input, prev_state)` as input and passes the input
  through each internal module in the order they were presented, using
  elements from `prev_state` as necessary for internal RNN cores.

  >>> deep_rnn = snt.DeepRNN([
  ...     snt.LSTM(hidden_size=16),
  ...     snt.LSTM(hidden_size=16),
  ... ])

  Note that the state of a `DeepRNN` is always a tuple, which will contain
  the same number of elements as there are internal RNN cores. If no
  internal modules are RNN cores, the state of the DeepRNN as a whole is
  an empty tuple.

  Wrapping non-recurrent modules into a DeepRNN can be useful to produce
  something API compatible with a "real" recurrent module, simplifying
  code that handles the cores.
  """

  # TODO(slebedev): currently called `layers` to be in-sync with `Sequential`.
  def __init__(self, layers, name=None):
    super(DeepRNN, self).__init__(layers, skip_connections=False, name=name)


def deep_rnn_with_skip_connections(
    layers,
    concat_final_output=True,
    name="deep_rnn_with_skip_connections"):
  """Construct a `DeepRNN` with skip connections.

  Skip connections alter the dependency structure within a `DeepRNN`.
  Specifically, input to the i-th layer (i > 0) is given by a
  concatenation of the core's inputs and the outputs of the (i-1)-th layer.

    outputs0, ... = layers[0](inputs, ...)
    outputs1, ... = layers[1](tf.concat([inputs, outputs0], axis=1], ...)
    outputs2, ... = layers[2](tf.concat([inputs, outputs1], axis=1], ...)
    ...

  This allows the layers to learn decoupled features.

  Args:
    layers: A list of `RNNCore`s.
    concat_final_output: If enabled (default), the outputs of the core
      is a concatenation of the outputs of all intermediate layers;
      otherwise, only the outputs of the final layer, i.e. that of
      `layers[-1]`, are returned.
    name: Name of the module.

  Returns:
    A `DeepRNN` with skip connections.

  Raises:
    ValueError: If any of the layers is not an `RNNCore`.
  """
  if not all(isinstance(l, RNNCore) for l in layers):
    raise ValueError(
        "deep_rnn_with_skip_connections requires all layers to be "
        "instances of RNNCore")

  return _LegacyDeepRNN(
      layers,
      skip_connections=True,
      concat_final_output_if_skip=concat_final_output,
      name=name)


class _ResidualWrapper(RNNCore):
  """Residual connection wrapper for a base RNN core.

  The output of the wrapper is the sum of the outputs of the base core
  with its inputs.
  """

  def __init__(self, base_core):
    super(_ResidualWrapper, self).__init__(name=base_core.name + "_residual")
    self._base_core = base_core

  def __call__(self, inputs, prev_state):
    """See base class."""
    outputs, next_state = self._base_core(inputs, prev_state)
    residual = nest.map_structure(lambda i, o: i + o, inputs, outputs)
    return residual, next_state

  def initial_state(self, batch_size, **kwargs):
    return self._base_core.initial_state(batch_size, **kwargs)


def deep_rnn_with_residual_connections(
    layers,
    name="deep_rnn_with_residual_connections"):
  """Construct a `DeepRNN` with residual connections.

  Residual connections alter the dependency structure in a `DeepRNN`.
  Specifically, the input to the i-th intermediate layer is a sum of
  the original core's inputs and the outputs of all the preceding
  layers (<i).

    outputs0, ... = layers[0](inputs, ...)
    outputs0 += inputs
    outputs1, ... = layers[1](outputs0, ...)
    outputs1 += outputs0
    outputs2, ... = layers[2](outputs1, ...)
    outputs2 += outputs1
    ...

  This allows the layers to learn specialized features that compose
  incrementally.

  Args:
    layers: A list of `RNNCore`s.
    name: Name of the module.

  Returns:
    A `DeepRNN` with residual connections.

  Raises:
    ValueError: If any of the layers is not an `RNNCore`.
  """

  if not all(isinstance(l, RNNCore) for l in layers):
    raise ValueError(
        "deep_rnn_with_residual_connections requires all layers to be "
        "instances of RNNCore")

  return _LegacyDeepRNN(
      [_ResidualWrapper(l) for l in layers],
      skip_connections=False,
      name=name)


LSTMState = collections.namedtuple("LSTMState", ["hidden", "cell"])


class LSTM(RNNCore):
  """Long short-term memory (LSTM) RNN core.

  The implementation is based on: http://arxiv.org/abs/1409.2329.
  Given x_t and the previous state (h_{t-1}, c_{t-1}) the core computes

    i_t = sigm(W_{ii} x_t + W_{hi} h_{t-1} + b_i)
    f_t = sigm(W_{if} x_t + W_{hf} h_{t-1} + b_f)
    g_t = tanh(W_{ig} x_t + W_{hg} h_{t-1} + b_g)
    o_t = sigm(W_{io} x_t + W_{ho} h_{t-1} + b_o)
    c_t = f_t c_{t-1} + i_t g_t
    h_t = o_t tanh(c_t)

  where i_t, f_t, o_t are input, forget and output gate activations,
  and g_t is a vector of cell updates.

  Following http://proceedings.mlr.press/v37/jozefowicz15.pdf we add a
  constant `forget_bias` (defaults to 1.0) to b_f in order to reduce
  the scale of forgetting in the beginning of the training.

  #### Recurrent projections

  Hidden state could be projected (via the `project_size` parameter)
  to reduce the number of parameters and speed up computation. For more
  details see https://arxiv.org/abs/1402.1128.

  Variables:
    w_i: input-to-hidden weights W_{ii}, W_{if}, W_{ig} and W_{io}
      concatenated into a Tensor of shape [input_size, 3 * hidden_size].
    w_h: hidden-to-hidden weights W_{hi}, W_{hf}, W_{hg} and W_{ho}
      concatenated into a Tensor of shape [hidden_size, 3 * hidden_size].
    b: biases b_i, b_f, b_g and b_o concatenated into a Tensor of shape
      [3 * hidden_size].
  """

  def __init__(
      self,
      hidden_size,
      projection_size=None,
      projection_init=None,
      w_i_init=None,
      w_h_init=None,
      b_init=None,
      forget_bias=1.0,
      dtype=tf.float32,
      name=None):
    """Construct an LSTM.

    Args:
      hidden_size: Hidden layer size.
      projection_size: Optional int; if set, then the hidden state is
        projected to this size via a trainable projection matrix.
      projection_init: Optional initializer for the projection matrix.
        Defaults to `TruncatedNormal` with a standard deviation of
        `1 / sqrt(hidden_size).
      w_i_init: Optional initializer for the input-to-hidden weights.
        Defaults to `TruncatedNormal` with a standard deviation of
        `1 / sqrt(input_size).
      w_h_init: Optional initializer for the hidden-to-hidden weights.
        Defaults to `TruncatedNormal` with a standard deviation of
        `1 / sqrt(hidden_size).
      b_init: Optional initializer for the biases. Defaults to
        `Zeros`.
      forget_bias: Optional float to add to the bias of the forget gate
        after initialization.
      dtype: Optional `tf.DType` of the core's variables. Defaults to
        `tf.float32`.
      name: Name of the module.
    """
    super(LSTM, self).__init__(name)
    self._hidden_size = hidden_size
    self._projection_size = projection_size
    self._eff_hidden_size = self._projection_size or self._hidden_size
    self._projection_init = projection_init
    if projection_size is None and projection_init is not None:
      raise ValueError(
          "projection_init must be None when projection is not used")

    self._w_i_init = w_i_init
    self._w_h_init = w_h_init
    self._b_init = b_init or initializers.Zeros()
    self._forget_bias = forget_bias
    self._dtype = dtype

  def __call__(self, inputs, prev_state):
    """See base class."""
    self._create_parameters(inputs)

    gates_x = tf.matmul(inputs, self._w_i)
    gates_h = tf.matmul(prev_state.hidden, self._w_h)
    gates = gates_x + gates_h + self.b

    # i = input, f = forget, g = cell updates, o = output.
    i, f, g, o = tf.split(gates, num_or_size_splits=4, axis=1)

    next_cell = tf.sigmoid(f) * prev_state.cell
    next_cell += tf.sigmoid(i) * tf.tanh(g)
    next_hidden = tf.sigmoid(o) * tf.tanh(next_cell)

    if self._projection_size is not None:
      next_hidden = tf.matmul(next_hidden, self.projection)

    return next_hidden, LSTMState(hidden=next_hidden, cell=next_cell)

  def initial_state(self, batch_size):
    """See base class."""
    return LSTMState(
        hidden=tf.zeros(
            [batch_size, self._eff_hidden_size],
            dtype=self._dtype),
        cell=tf.zeros([batch_size, self._hidden_size], dtype=self._dtype))

  @property
  def input_to_hidden(self):
    return self._w_i

  @property
  def hidden_to_hidden(self):
    return self._w_h

  @once.once
  def _create_parameters(self, inputs):
    utils.assert_rank(inputs, 2)
    input_size = tf.shape(inputs)[1]
    dtype = _check_inputs_dtype(inputs, self._dtype)

    w_i_init = self._w_i_init or initializers.TruncatedNormal(
        stddev=1.0 / tf.sqrt(tf.cast(input_size, dtype)))
    w_h_init = self._w_h_init or initializers.TruncatedNormal(
        stddev=1.0 / tf.sqrt(tf.constant(self._eff_hidden_size, dtype=dtype)))
    self._w_i = tf.Variable(
        w_i_init([input_size, 4 * self._hidden_size], dtype),
        name="w_i")
    self._w_h = tf.Variable(
        w_h_init([self._eff_hidden_size, 4 * self._hidden_size], dtype),
        name="w_h")

    i, f, g, o = tf.split(
        self._b_init([4 * self._hidden_size], dtype),
        num_or_size_splits=4)
    f += self._forget_bias
    self.b = tf.Variable(tf.concat([i, f, g, o], axis=0), name="b")

    if self._projection_size is None:
      self.projection = None
    else:
      projection_init = self._projection_init
      if projection_init is None:
        projection_init = initializers.TruncatedNormal(
            stddev=1.0 / tf.sqrt(tf.constant(self._hidden_size, dtype=dtype)))
      self.projection = tf.Variable(
          projection_init([self._hidden_size, self._projection_size], dtype),
          name="projection")


class CuDNNLSTM(RNNCore):
  """Long short-term memory (LSTM) RNN implemented using CuDNN-RNN.

  Unlike `LSTM` this core operates on the whole batch of sequences at
  once, i.e. the expected shape of `inputs` is
  [num_steps, batch_size, input_size].
  """

  def __init__(
      self,
      hidden_size,
      w_i_init=None,
      w_h_init=None,
      b_init=None,
      forget_bias=1.0,
      dtype=tf.float32,
      name=None):
    """Construct an LSTM.

    Args:
      hidden_size: Hidden layer size.
      w_i_init: Optional initializer for the input-to-hidden weights.
        Defaults to `TruncatedNormal` with a standard deviation of
        `1 / sqrt(input_size).
      w_h_init: Optional initializer for the hidden-to-hidden weights.
        Defaults to `TruncatedNormal` with a standard deviation of
        `1 / sqrt(hidden_size).
      b_init: Optional initializer for the biases. Defaults to
        `Zeros`.
      forget_bias: Optional float to add to the bias of the forget gate
        after initialization.
      dtype: Optional `tf.DType` of the core's variables. Defaults to
        `tf.float32`.
      name: Name of the module.
    """
    super(CuDNNLSTM, self).__init__(name)
    self._hidden_size = hidden_size
    self._w_i_init = w_i_init
    self._w_h_init = w_h_init
    self._b_init = b_init or initializers.Zeros()
    self._forget_bias = forget_bias
    self._dtype = dtype

  def __call__(self, inputs, prev_state):
    """See base class."""
    self._create_parameters(inputs)

    # TODO(slebedev): consider allocating a single parameter Tensor.
    # This will remove the need for tf.transpose and tf.concat and
    # will likely result in a significant speedup. On the downside,
    # checkpoints of `CuDNNLSTM` incompatible with that of `LSTM`.
    b_h_zero = tf.zeros([self._hidden_size])
    outputs, next_hidden, next_cell, _ = tf.raw_ops.CudnnRNN(
        input=inputs,
        input_h=tf.expand_dims(prev_state.hidden, axis=0),
        input_c=tf.expand_dims(prev_state.cell, axis=0),
        params=tf.concat([
            tf.reshape(tf.transpose(self._w_i), [-1]),
            tf.reshape(tf.transpose(self._w_h), [-1]),
            # CuDNN has two sets of biases: b_i and b_h, zero-out b_h.
            self.b,
            b_h_zero,
            b_h_zero,
            b_h_zero,
            b_h_zero
        ], axis=0),
        rnn_mode="lstm")

    return outputs, LSTMState(hidden=next_hidden, cell=next_cell)

  def initial_state(self, batch_size):
    """See base class."""
    return LSTMState(
        hidden=tf.zeros([batch_size, self._hidden_size], dtype=self._dtype),
        cell=tf.zeros([batch_size, self._hidden_size], dtype=self._dtype))

  @property
  def input_to_hidden(self):
    return self._w_i

  @property
  def hidden_to_hidden(self):
    return self._w_h

  @once.once
  def _create_parameters(self, inputs):
    utils.assert_rank(inputs, 3)  # [num_steps, batch_size, input_size].
    input_size = tf.shape(inputs)[2]
    dtype = _check_inputs_dtype(inputs, self._dtype)

    w_i_init = self._w_i_init or initializers.TruncatedNormal(
        stddev=1.0 / tf.sqrt(tf.cast(input_size, dtype)))
    w_h_init = self._w_h_init or initializers.TruncatedNormal(
        stddev=1.0 / tf.sqrt(tf.constant(self._hidden_size, dtype=dtype)))
    self._w_i = tf.Variable(
        w_i_init([input_size, 4 * self._hidden_size], dtype),
        name="w_i")
    self._w_h = tf.Variable(
        w_h_init([self._hidden_size, 4 * self._hidden_size], dtype),
        name="w_h")

    i, f, g, o = tf.split(
        self._b_init([4 * self._hidden_size], dtype),
        num_or_size_splits=4)
    f += self._forget_bias
    self.b = tf.Variable(tf.concat([i, f, g, o], axis=0), name="b")


class _RecurrentDropoutWrapper(RNNCore):
  """Recurrent dropout wrapper for a base RNN core.

  The wrapper drops the previous state of the base core according to
  dropout `rates`. Specifically, dropout is only applied if the rate
  corresponding to the state element is not `None`. Dropout masks
  are sampled in `initial_state` of the wrapper.

  This class is not intended to be used directly. See
  `lstm_with_recurrent_dropout`.
  """

  def __init__(self, base_core, rates, seed=None):
    """Wrap a given base RNN core.

    Args:
      base_core: The RNNCore to be wrapped
      rates: Recurrent dropout probabilities. The structure should
        match that of `base_core.initial_state`.
      seed: Optional int; seed passed to `tf.nn.dropout`.
    """
    super(_RecurrentDropoutWrapper, self).__init__(
        name=base_core.name + "_recurrent_dropout")
    self._base_core = base_core
    self._rates = rates
    self._seed = seed

  def __call__(self, inputs, prev_state):
    prev_core_state, dropout_masks = prev_state
    prev_core_state = nest.map_structure(
        lambda s, mask: s if mask is None else s * mask,
        prev_core_state,
        dropout_masks)
    output, next_core_state = self._base_core(inputs, prev_core_state)
    return output, (next_core_state, dropout_masks)

  def initial_state(self, batch_size, **kwargs):
    core_initial_state = self._base_core.initial_state(batch_size, **kwargs)

    def maybe_dropout(s, rate):
      if rate is None:
        return None
      else:
        return tf.nn.dropout(tf.ones_like(s), rate=rate, seed=self._seed)

    dropout_masks = nest.map_structure(
        maybe_dropout,
        core_initial_state,
        self._rates)
    return core_initial_state, dropout_masks


def lstm_with_recurrent_dropout(
    hidden_size,
    dropout=0.5,
    seed=None,
    **kwargs):
  """Construct an LSTM with recurrent dropout.

  The implementation is based on https://arxiv.org/abs/1512.05287.
  Dropout is applied on the previous hidden state h_{t-1} during the
  computation of gate activations

    i_t = sigm(W_{ii} x_t + W_{hi} d(h_{t-1}) + b_i)
    f_t = sigm(W_{if} x_t + W_{hf} d(h_{t-1}) + b_f)
    g_t = tanh(W_{ig} x_t + W_{hg} d(h_{t-1}) + b_g)
    o_t = sigm(W_{io} x_t + W_{ho} d(h_{t-1}) + b_o)

  Args:
    hidden_size: Hidden layer size.
    dropout: Dropout probability.
    seed: Optional int; seed passed to `tf.nn.dropout`.
    **kwargs: Optional keyword arguments to pass to the `LSTM` constructor.

  Returns:
    train_lstm: an `LSTM` with recurrent dropout enabled for training.
    test_lstm: the same as `train_lstm` but without recurrent dropout.

  Raises:
    ValueError: If `dropout` is not in `[0, 1)`.
  """
  if dropout < 0 or dropout >= 1:
    raise ValueError(
        "dropout must be in the range [0, 1), got {}".format(dropout))

  lstm = LSTM(hidden_size, **kwargs)
  rate = LSTMState(hidden=dropout, cell=None)
  return _RecurrentDropoutWrapper(lstm, rate, seed), lstm


class _ConvNDLSTM(RNNCore):
  """Convolutional LSTM.

  The implementation is based on: https://arxiv.org/abs/1506.04214.
  Given x_t and the previous state (h_{t-1}, c_{t-1}) the core computes

    i_t = sigm(W_{ii} * x_t + W_{hi} * h_{t-1} + b_i)
    f_t = sigm(W_{if} * x_t + W_{hf} * h_{t-1} + b_f)
    g_t = tanh(W_{ig} * x_t + W_{hg} * h_{t-1} + b_g)
    o_t = sigm(W_{io} * x_t + W_{ho} * h_{t-1} + b_o)
    c_t = f_t c_{t-1} + i_t g_t
    h_t = o_t tanh(c_t)

  where * denotes the convolution operator; i_t, f_t, o_t are input,
  forget and output gate activations, and g_t is a vector of cell
  updates.

  Following http://proceedings.mlr.press/v37/jozefowicz15.pdf we add a
  constant `forget_bias` (defaults to 1.0) to b_f in order to reduce
  the scale of forgetting in the beginning of the training.

  Variables:
    input_to_hidden/w: convolution weights W_{ii}, W_{if}, W_{ig} and
       W_{io} concatenated into a single Tensor of shape
       [kernel_shape*, input_channels, 4 * output_channels] where
       `kernel_shape` is repeated `num_spatial_dims` times.
    hidden_to_hidden/w: convolution weights W_{hi}, W_{hf}, W_{hg} and
       W_{ho} concatenated into a single Tensor of shape
       [kernel_shape*, input_channels, 4 * output_channels] where
       `kernel_shape` is repeated `num_spatial_dims` times.
    b: biases b_i, b_f, b_g and b_o concatenated into a Tensor of shape
      [4 * output_channels].
  """

  def __init__(
      self,
      num_spatial_dims,
      input_shape,
      output_channels,
      kernel_shape,
      data_format=None,
      w_i_init=None,
      w_h_init=None,
      b_init=None,
      forget_bias=1.0,
      dtype=tf.float32,
      name=None):
    """Constructs a convolutional LSTM.

    Args:
      num_spatial_dims: Number of spatial dimensions of the input.
      input_shape: Shape of the inputs excluding batch size.
      output_channels: Number of output channels.
      kernel_shape: Sequence of kernel sizes (of length num_spatial_dims),
        or an int. `kernel_shape` will be expanded to define a kernel
        size in all dimensions.
      data_format: The data format of the input.
      w_i_init: Optional initializer for the input-to-hidden convolution
        weights. Defaults to `TruncatedNormal` with a standard deviation of
        `1 / sqrt(kernel_shape**num_spatial_dims * input_channels)`.
      w_h_init: Optional initializer for the hidden-to-hidden convolution
        weights. Defaults to `TruncatedNormal` with a standard deviation of
        `1 / sqrt(kernel_shape**num_spatial_dims * input_channels)`.
      b_init: Optional initializer for the biases. Defaults to zeros.
      forget_bias: Optional float to add to the bias of the forget gate
        after initialization.
      dtype: Optional `tf.DType` of the core's variables. Defaults to
        `tf.float32`.
      name: Name of the module.
    """
    super(_ConvNDLSTM, self).__init__(name)
    self._num_spatial_dims = num_spatial_dims
    self._input_shape = list(input_shape)
    self._channel_index = 1 if data_format.startswith("NC") else -1
    self._output_channels = output_channels
    self._b_init = b_init or initializers.Zeros()
    self._forget_bias = forget_bias
    self._dtype = dtype

    self._input_to_hidden = conv.ConvND(
        self._num_spatial_dims,
        output_channels=4 * output_channels,
        kernel_shape=kernel_shape,
        padding="SAME",
        with_bias=False,
        w_init=w_i_init,
        data_format=data_format,
        name="input_to_hidden")
    self._hidden_to_hidden = conv.ConvND(
        self._num_spatial_dims,
        output_channels=4 * output_channels,
        kernel_shape=kernel_shape,
        padding="SAME",
        with_bias=False,
        w_init=w_h_init,
        data_format=data_format,
        name="hidden_to_hidden")

  def __call__(self, inputs, prev_state):
    """See base class."""
    self._create_parameters(inputs)

    gates = self._input_to_hidden(inputs)
    gates += self._hidden_to_hidden(prev_state.hidden)
    gates += self.b

    # i = input, f = forget, g = cell updates, o = output.
    i, f, g, o = tf.split(
        gates,
        num_or_size_splits=4,
        axis=self._num_spatial_dims + 1)

    next_cell = tf.sigmoid(f) * prev_state.cell
    next_cell += tf.sigmoid(i) * tf.tanh(g)
    next_hidden = tf.sigmoid(o) * tf.tanh(next_cell)
    return next_hidden, LSTMState(hidden=next_hidden, cell=next_cell)

  @property
  def input_to_hidden(self):
    return self._input_to_hidden.w

  @property
  def hidden_to_hidden(self):
    return self._hidden_to_hidden.w

  def initial_state(self, batch_size):
    """See base class."""
    shape = list(self._input_shape)
    shape[self._channel_index] = self._output_channels
    shape = [batch_size] + shape
    return LSTMState(
        hidden=tf.zeros(shape, dtype=self._dtype),
        cell=tf.zeros(shape, dtype=self._dtype))

  @once.once
  def _create_parameters(self, inputs):
    dtype = _check_inputs_dtype(inputs, self._dtype)
    i, f, g, o = tf.split(
        self._b_init([4 * self._output_channels], dtype),
        num_or_size_splits=4)
    f += self._forget_bias
    self.b = tf.Variable(tf.concat([i, f, g, o], axis=0), name="b")


class Conv1DLSTM(_ConvNDLSTM):
  """See `_ConvNDLSTM`."""

  def __init__(
      self,
      input_shape,
      output_channels,
      kernel_shape,
      data_format="NWC",
      w_i_init=None,
      w_h_init=None,
      b_init=None,
      forget_bias=1.0,
      dtype=tf.float32,
      name=None):
    super(Conv1DLSTM, self).__init__(
        num_spatial_dims=1,
        input_shape=input_shape,
        output_channels=output_channels,
        kernel_shape=kernel_shape,
        data_format=data_format,
        w_i_init=w_i_init,
        w_h_init=w_h_init,
        b_init=b_init,
        forget_bias=forget_bias,
        dtype=dtype,
        name=name)


class Conv2DLSTM(_ConvNDLSTM):
  """See `_ConvNDLSTM`."""

  def __init__(
      self,
      input_shape,
      output_channels,
      kernel_shape,
      data_format="NHWC",
      w_i_init=None,
      w_h_init=None,
      b_init=None,
      forget_bias=1.0,
      dtype=tf.float32,
      name=None):
    super(Conv2DLSTM, self).__init__(
        num_spatial_dims=2,
        input_shape=input_shape,
        output_channels=output_channels,
        kernel_shape=kernel_shape,
        data_format=data_format,
        w_i_init=w_i_init,
        w_h_init=w_h_init,
        b_init=b_init,
        forget_bias=forget_bias,
        dtype=dtype,
        name=name)


class Conv3DLSTM(_ConvNDLSTM):
  """See `_ConvNDLSTM`."""

  def __init__(
      self,
      input_shape,
      output_channels,
      kernel_shape,
      data_format="NDHWC",
      w_i_init=None,
      w_h_init=None,
      b_init=None,
      forget_bias=1.0,
      dtype=tf.float32,
      name=None):
    super(Conv3DLSTM, self).__init__(
        num_spatial_dims=3,
        input_shape=input_shape,
        output_channels=output_channels,
        kernel_shape=kernel_shape,
        data_format=data_format,
        w_i_init=w_i_init,
        w_h_init=w_h_init,
        b_init=b_init,
        forget_bias=forget_bias,
        dtype=dtype,
        name=name)


class GRU(RNNCore):
  """Gated recurrent unit (GRU) RNN core.

  The implementation is based on: https://arxiv.org/abs/1412.3555.
  Given x_t and previous hidden state h_{t-1} the core computes

      z_t = sigm(W_{iz} x_t + W_{hz} h_{t-1} + b_z)
      r_t = sigm(W_{ir} x_t + W_{hr} h_{t-1} + b_r)
      a_t = tanh(W_{ia} x_t + W_{ha} (r_t h_{t-1}) + b_a)
      h_t = (1 - z_t) h_{t-1} + z_t a_t

  where z_t and r_t are reset and update gates.

  Variables:
    w_i: input-to-hidden weights W_{iz}, W_{ir} and W_{ia} concatenated
      into a Tensor of shape [input_size, 3 * hidden_size].
    w_h: hidden-to-hidden weights W_{hz}, W_{hr} and W_{ha} concatenated
      into a Tensor of shape [hidden_size, 3 * hidden_size].
    b: biases b_z, b_r and b_a concatenated into a Tensor of shape
      [3 * hidden_size].
  """

  def __init__(
      self,
      hidden_size,
      w_i_init=None,
      w_h_init=None,
      b_init=None,
      dtype=tf.float32,
      name=None):
    """Construct a GRU.

    Args:
      hidden_size: Hidden layer size.
      w_i_init: Optional initializer for the input-to-hidden weights.
        Defaults to Glorot uniform initializer.
      w_h_init: Optional initializer for the hidden-to-hidden weights.
        Defaults to Glorot uniform initializer.
      b_init: Optional initializer for the biases. Defaults to `Zeros`.
      dtype: Optional `tf.DType` of the core's variables. Defaults to
        `tf.float32`.
      name: Name of the module.
    """
    super(GRU, self).__init__(name)
    self._hidden_size = hidden_size
    glorot_uniform = initializers.VarianceScaling(
        mode="fan_avg",
        distribution="uniform")
    self._w_i_init = w_i_init or glorot_uniform
    self._w_h_init = w_h_init or glorot_uniform
    self._b_init = b_init or initializers.Zeros()
    self._dtype = dtype

  def __call__(self, inputs, prev_state):
    """See base class."""
    self._create_parameters(inputs)

    gates_x = tf.matmul(inputs, self._w_i)
    zr_idx = slice(2 * self._hidden_size)
    zr_x = gates_x[:, zr_idx]
    zr_h = tf.matmul(prev_state, self._w_h[:, zr_idx])
    zr = zr_x + zr_h + self.b[zr_idx]
    z, r = tf.split(tf.sigmoid(zr), num_or_size_splits=2, axis=1)

    a_idx = slice(2 * self._hidden_size, 3 * self._hidden_size)
    a_x = gates_x[:, a_idx]
    a_h = tf.matmul(r * prev_state, self._w_h[:, a_idx])
    a = tf.tanh(a_x + a_h + self.b[a_idx])

    next_state = (1 - z) * prev_state + z * a
    return next_state, next_state

  def initial_state(self, batch_size):
    """See base class."""
    return tf.zeros([batch_size, self._hidden_size], dtype=self._dtype)

  @property
  def input_to_hidden(self):
    return self._w_i

  @property
  def hidden_to_hidden(self):
    return self._w_h

  @once.once
  def _create_parameters(self, inputs):
    utils.assert_rank(inputs, 2)
    input_size = tf.shape(inputs)[1]
    dtype = _check_inputs_dtype(inputs, self._dtype)
    self._w_i = tf.Variable(
        self._w_i_init([input_size, 3 * self._hidden_size], dtype),
        name="w_i")
    self._w_h = tf.Variable(
        self._w_h_init([self._hidden_size, 3 * self._hidden_size], dtype),
        name="w_h")
    self.b = tf.Variable(
        self._b_init([3 * self._hidden_size], dtype),
        name="b")


class CuDNNGRU(RNNCore):
  """Gated recurrent unit (GRU) RNN core implemented using CuDNN-RNN.

  The (CuDNN) implementation is based on https://arxiv.org/abs/1406.1078
  and differs from `GRU` in the way a_t and h_t are computed:

      a_t = tanh(W_{ia} x_t + r_t (W_{ha} h_{t-1}) + b_a)
      h_t = (1 - z_t) a_t + z_t h_{t-1}

  Unlike `GRU` this core operates on the whole batch of sequences at
  once, i.e. the expected shape of `inputs` is
  [num_steps, batch_size, input_size].
  """

  def __init__(
      self,
      hidden_size,
      w_i_init=None,
      w_h_init=None,
      b_init=None,
      dtype=tf.float32,
      name=None):
    """Construct a GRU.

    Args:
      hidden_size: Hidden layer size.
      w_i_init: Optional initializer for the input-to-hidden weights.
        Defaults to Glorot uniform initializer.
      w_h_init: Optional initializer for the hidden-to-hidden weights.
        Defaults to Glorot uniform initializer.
      b_init: Optional initializer for the biases. Defaults to `Zeros`.
      dtype: Optional `tf.DType` of the core's variables. Defaults to
        `tf.float32`.
      name: Name of the module.
    """
    super(CuDNNGRU, self).__init__(name)
    self._hidden_size = hidden_size
    glorot_uniform = initializers.VarianceScaling(
        mode="fan_avg",
        distribution="uniform")
    self._w_i_init = w_i_init or glorot_uniform
    self._w_h_init = w_h_init or glorot_uniform
    self._b_init = b_init or initializers.Zeros()
    self._dtype = dtype

  def __call__(self, inputs, prev_state):
    """See base class."""
    self._create_parameters(inputs)

    # TODO(slebedev): consider allocating a single parameter Tensor.
    # This will remove the need for tf.transpose and tf.concat and
    # will likely result in a significant speedup. On the downside,
    # checkpoints of `CuDNNGRU` incompatible with that of `GRU`.
    # CuDNN orders the gates as r, z (instead of z, r).
    w_iz, w_ir, w_ia = tf.split(self._w_i, num_or_size_splits=3, axis=1)
    w_hz, w_hr, w_ha = tf.split(self._w_h, num_or_size_splits=3, axis=1)
    b_z, b_r, b_a = tf.split(self.b, num_or_size_splits=3)
    b_h_zero = tf.zeros([self._hidden_size])
    outputs, next_hidden, _, _ = tf.raw_ops.CudnnRNN(
        input=inputs,
        input_h=tf.expand_dims(prev_state, axis=0),
        input_c=0,
        params=tf.concat([
            tf.reshape(tf.transpose(w_ir), [-1]),
            tf.reshape(tf.transpose(w_iz), [-1]),
            tf.reshape(tf.transpose(w_ia), [-1]),
            tf.reshape(tf.transpose(w_hr), [-1]),
            tf.reshape(tf.transpose(w_hz), [-1]),
            tf.reshape(tf.transpose(w_ha), [-1]),
            # CuDNN has two sets of biases: b_i and b_h, zero-out b_h
            # to match the definition in `GRU`.
            b_r,
            b_z,
            b_a,
            b_h_zero,
            b_h_zero,
            b_h_zero,
        ], axis=0),
        rnn_mode="gru")

    return outputs, next_hidden

  @property
  def input_to_hidden(self):
    return self._w_i

  @property
  def hidden_to_hidden(self):
    return self._w_h

  def initial_state(self, batch_size):
    """See base class."""
    return tf.zeros([batch_size, self._hidden_size], dtype=self._dtype)

  @once.once
  def _create_parameters(self, inputs):
    utils.assert_rank(inputs, 3)  # [num_steps, batch_size, input_size].
    input_size = tf.shape(inputs)[2]
    dtype = _check_inputs_dtype(inputs, self._dtype)
    self._w_i = tf.Variable(
        self._w_i_init([input_size, 3 * self._hidden_size], dtype),
        name="w_i")
    self._w_h = tf.Variable(
        self._w_h_init([self._hidden_size, 3 * self._hidden_size], dtype),
        name="w_h")
    self.b = tf.Variable(
        self._b_init([3 * self._hidden_size], dtype),
        name="b")


def _check_inputs_dtype(inputs, expected_dtype):
  if inputs.dtype is not expected_dtype:
    raise TypeError("inputs must have dtype {!r}, got {!r}"
                    .format(expected_dtype, inputs.dtype))
  return expected_dtype
