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

import abc
import collections
import functools
from typing import Optional, Sequence, Tuple, Union
import uuid

from sonnet.src import base
from sonnet.src import conv
from sonnet.src import initializers
from sonnet.src import linear
from sonnet.src import once
from sonnet.src import types
from sonnet.src import utils
import tensorflow.compat.v1 as tf1
import tensorflow as tf
import tree

# pylint: disable=g-direct-tensorflow-import
# Required for specializing `UnrolledLSTM` per device.
from tensorflow.python import context as context_lib
from tensorflow.python.eager import function as function_lib
# pylint: enable=g-direct-tensorflow-import


class RNNCore(base.Module, metaclass=abc.ABCMeta):
  """Base class for Recurrent Neural Network cores.

  This class defines the basic functionality that every core should
  implement: :meth:`initial_state`, used to construct an example of the
  core state; and :meth:`__call__` which applies the core parameterized
  by a previous state to an input.

  Cores are typically used with :func:`dynamic_unroll` and
  :func:`static_unroll` to iteratively construct an output sequence from
  the given input sequence.
  """

  @abc.abstractmethod
  def __call__(self, inputs: types.TensorNest, prev_state):
    """Performs one step of an RNN.

    Args:
      inputs: An arbitrarily nested structure of shape [B, ...] where B is the
        batch size.
      prev_state: Previous core state.

    Returns:
      A tuple with two elements:
      * **outputs** - An arbitrarily nested structure of shape [B, ...].
        Dimensions following the batch size could be different from that
        of `inputs`.
      * **next_state** - Next core state, must be of the same shape as the
        previous one.
    """

  @abc.abstractmethod
  def initial_state(self, batch_size: types.IntegerLike, **kwargs):
    """Constructs an initial state for this core.

    Args:
      batch_size: An int or an integral scalar tensor representing batch size.
      **kwargs: Optional keyword arguments.

    Returns:
      Arbitrarily nested initial state for this core.
    """


class UnrolledRNN(base.Module, metaclass=abc.ABCMeta):
  """Base class for unrolled Recurrent Neural Networks.

  This class is a generalization of :class:`RNNCore` which operates on
  an input sequence as opposed to a single time step.
  """

  @abc.abstractmethod
  def __call__(self, input_sequence: types.TensorNest,
               initial_state: types.TensorNest):
    """Apply this RNN to the input sequence.

    Args:
      input_sequence: An arbitrarily nested structure of shape ``[T, B, ...]``
        where ``T`` is the number of time steps and B is the batch size.
      initial_state: Initial RNN state.

    Returns:
      A tuple with two elements:
        * **output_sequence** - An arbitrarily nested structure of tensors of
          shape ``[T, B, ...]``. Dimensions following the batch size could be
          different from that of the ``input_sequence``.
        * **final_state** - Final RNN state, must be of the same shape as the
          initial one.
    """

  @abc.abstractmethod
  def initial_state(self, batch_size: types.IntegerLike, **kwargs):
    """Construct an initial state for this RNN.

    Args:
      batch_size: An int or an integral scalar tensor representing batch size.
      **kwargs: Optional keyword arguments.

    Returns:
      Arbitrarily nested initial state for this RNN.
    """


class TrainableState(base.Module):
  """Trainable state for an :class:`RNNCore`.

  The state can be constructed manually from a nest of initial values::

      >>> state = snt.TrainableState((tf.zeros([16]), tf.zeros([16])))

  or automatically for a given :class:`RNNCore`::

      >>> core = snt.LSTM(hidden_size=16)
      >>> state = snt.TrainableState.for_core(core)
  """

  @classmethod
  def for_core(cls,
               core: RNNCore,
               mask: Optional[types.TensorNest] = None,
               name: Optional[str] = None):
    """Constructs a trainable state for a given :class:`RNNCore`.

    Args:
      core: An :class:`RNNCore` to construct the state for.
      mask: Optional boolean mask of the same structure as the initial state of
        `core` specifying which components should be trainable. If not given,
        the whole state is considered trainable.
      name: Name of the module.

    Returns:
      A `TrainableState`.
    """
    initial_values = tree.map_structure(lambda s: tf.squeeze(s, axis=0),
                                        core.initial_state(batch_size=1))
    return cls(initial_values, mask, name)

  def __init__(self,
               initial_values: types.TensorNest,
               mask: Optional[types.TensorNest] = None,
               name: Optional[str] = None):
    """Constructs a trainable state from initial values.

    Args:
      initial_values: Arbitrarily nested initial values for the state.
      mask: Optional boolean mask of the same structure as ``initial_values``
        specifying which components should be trainable. If not given, the whole
        state is considered trainable.
      name: Name of the module.
    """
    super().__init__(name)

    flat_initial_values = tree.flatten_with_path(initial_values)
    if mask is None:
      flat_mask = [True] * len(flat_initial_values)
    else:
      tree.assert_same_structure(initial_values, mask)
      flat_mask = tree.flatten(mask)

    flat_template = []
    for (path, initial_value), trainable in zip(flat_initial_values, flat_mask):
      # `"state"` is only used if initial_values is not nested.
      name = "/".join(map(str, path)) or "state"
      flat_template.append(
          tf.Variable(
              tf.expand_dims(initial_value, axis=0),
              trainable=trainable,
              name=name))

    self._template = tree.unflatten_as(initial_values, flat_template)

  def __call__(self, batch_size: int) -> types.TensorNest:
    """Returns a trainable state for the given batch size."""
    return tree.map_structure(
        lambda s: tf.tile(s, [batch_size] + [1] * (s.shape.rank - 1)),
        self._template)


def static_unroll(
    core: RNNCore,
    input_sequence: types.TensorNest,  # time-major.
    initial_state: types.TensorNest,
    sequence_length: Optional[types.IntegerLike] = None
) -> Tuple[types.TensorNest, types.TensorNest]:
  """Performs a static unroll of an RNN.

      >>> core = snt.LSTM(hidden_size=16)
      >>> batch_size = 3
      >>> input_sequence = tf.random.uniform([1, batch_size, 2])
      >>> output_sequence, final_state = snt.static_unroll(
      ...     core,
      ...     input_sequence,
      ...     core.initial_state(batch_size))

  An *unroll* corresponds to calling the core on each element of the
  input sequence in a loop, carrying the state through::

      state = initial_state
      for t in range(len(input_sequence)):
         outputs, state = core(input_sequence[t], state)

  A *static* unroll replaces a loop with its body repeated multiple
  times when executed inside :tf:`function`::

      state = initial_state
      outputs0, state = core(input_sequence[0], state)
      outputs1, state = core(input_sequence[1], state)
      outputs2, state = core(input_sequence[2], state)
      ...

  See :func:`dynamic_unroll` for a loop-preserving unroll function.

  Args:
    core: An :class:`RNNCore` to unroll.
    input_sequence: An arbitrarily nested structure of tensors of shape
      ``[T, B, ...]`` where ``T`` is the number of time steps, and ``B`` is
      the batch size.
    initial_state: An initial state of the given core.
    sequence_length: An optional tensor of shape ``[B]`` specifying the lengths
      of sequences within the (padded) batch.

  Returns:
    A tuple with two elements:
      * **output_sequence** - An arbitrarily nested structure of tensors
        of shape ``[T, B, ...]``. Dimensions following the batch size could
        be different from that of the ``input_sequence``.
      * **final_state** - Core state at time step ``T``.

  Raises:
    ValueError: If ``input_sequence`` is empty or its leading dimension is
      not known statically.
  """
  num_steps, input_tas = _unstack_input_sequence(input_sequence)
  if not isinstance(num_steps, int):
    raise ValueError(
        "input_sequence must have a statically known number of time steps")

  outputs = None
  state = initial_state
  output_accs = None
  for t in range(num_steps):
    outputs, state = _rnn_step(
        core,
        input_tas,
        sequence_length,
        t,
        prev_outputs=outputs,
        prev_state=state)
    if t == 0:
      output_accs = tree.map_structure(lambda o: _ListWrapper([o]), outputs)
    else:
      tree.map_structure(lambda acc, o: acc.data.append(o), output_accs,
                         outputs)

  output_sequence = tree.map_structure(lambda acc: tf.stack(acc.data),
                                       output_accs)
  return output_sequence, state


class _ListWrapper:
  """A wrapper hiding a list from `nest`.

  This allows to use `tree.map_structure` without recursing into the
  wrapped list.
  """

  __slots__ = ["data"]

  def __init__(self, data):
    self.data = data


# TODO(slebedev): core can be core_fn: Callable[[I, S], Tuple[O, S]].
# TODO(slebedev): explain sequence_length with ASCII art?
@utils.smart_autograph
def dynamic_unroll(
    core,
    input_sequence,  # time-major.
    initial_state,
    sequence_length=None,
    parallel_iterations=1,
    swap_memory=False):
  """Performs a dynamic unroll of an RNN.

      >>> core = snt.LSTM(hidden_size=16)
      >>> batch_size = 3
      >>> input_sequence = tf.random.uniform([1, batch_size, 2])
      >>> output_sequence, final_state = snt.dynamic_unroll(
      ...     core,
      ...     input_sequence,
      ...     core.initial_state(batch_size))

  An *unroll* corresponds to calling the core on each element of the
  input sequence in a loop, carrying the state through::

      state = initial_state
      for t in range(len(input_sequence)):
         outputs, state = core(input_sequence[t], state)

  A *dynamic* unroll preserves the loop structure when executed within
  :tf:`function`. See :func:`static_unroll` for an unroll function which
  replaces a loop with its body repeated multiple times.

  Args:
    core: An :class:`RNNCore` to unroll.
    input_sequence: An arbitrarily nested structure of tensors of shape
      ``[T, B, ...]`` where ``T`` is the number of time steps, and ``B`` is the
      batch size.
    initial_state: initial state of the given core.
    sequence_length: An optional tensor of shape ``[B]`` specifying the lengths
      of sequences within the (padded) batch.
    parallel_iterations: An optional ``int`` specifying the number of iterations
      to run in parallel. Those operations which do not have any temporal
      dependency and can be run in parallel, will be. This parameter trades off
      time for space. Values >> 1 use more memory but take less time, while
      smaller values use less memory but computations take longer. Defaults to
      1.
    swap_memory: Transparently swap the tensors produced in forward inference
      but needed for back prop from GPU to CPU. This allows training RNNs which
      would typically not fit on a single GPU, with very minimal (or no)
      performance penalty. Defaults to False.

  Returns:
    A tuple with two elements:
      * **output_sequence** - An arbitrarily nested structure of tensors
        of shape ``[T, B, ...]``. Dimensions following the batch size could
        be different from that of the ``input_sequence``.
      * **final_state** - Core state at time step ``T``.

  Raises:
    ValueError: If ``input_sequence`` is empty.
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
  output_tas = tree.map_structure(
      lambda o: tf.TensorArray(o.dtype, num_steps).write(0, o), outputs)

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
    output_tas = tree.map_structure(
        lambda ta, o, _t=t: ta.write(_t, o), output_tas, outputs)

  output_sequence = tree.map_structure(tf.TensorArray.stack, output_tas)
  return output_sequence, state


def _unstack_input_sequence(input_sequence):
  r"""Unstacks the input sequence into a nest of :tf:`TensorArray`\ s.

  This allows to traverse the input sequence using :tf:`TensorArray.read`
  instead of a slice, avoiding O(sliced tensor) slice gradient
  computation during the backwards pass.

  Args:
    input_sequence: See :func:`dynamic_unroll` or :func:`static_unroll`.

  Returns:
    num_steps: Number of steps in the input sequence.
    input_tas: An arbitrarily nested structure of :tf:`TensorArray`\ s of
      size ``num_steps``.

  Raises:
    ValueError: If tensors in ``input_sequence`` have inconsistent number
      of steps or the number of steps is 0.
  """
  flat_input_sequence = tree.flatten(input_sequence)
  all_num_steps = {i.shape[0] for i in flat_input_sequence}
  if len(all_num_steps) > 1:
    raise ValueError(
        "input_sequence tensors must have consistent number of time steps")
  [num_steps] = all_num_steps
  if num_steps == 0:
    raise ValueError("input_sequence must have at least a single time step")
  elif num_steps is None:
    # Number of steps is not known statically, fall back to dynamic shape.
    num_steps = tf.shape(flat_input_sequence[0])[0]
    # TODO(b/141910613): uncomment when the bug is fixed.
    # for i in flat_input_sequence[1:]:
    #   tf.debugging.assert_equal(
    #       tf.shape(i)[0], num_steps,
    #       "input_sequence tensors must have consistent number of time steps")

  input_tas = tree.map_structure(
      lambda i: tf.TensorArray(i.dtype, num_steps).unstack(i), input_sequence)
  return num_steps, input_tas


def _safe_where(condition, x, y):  # pylint: disable=g-doc-args
  """`tf.where` which allows scalar inputs."""
  if x.shape.rank == 0:
    # This is to match the `tf.nn.*_rnn` behavior. In general, we might
    # want to branch on `tf.reduce_all(condition)`.
    return y
  # TODO(tomhennigan) Broadcasting with SelectV2 is currently broken.
  return tf1.where(condition, x, y)


def _rnn_step(core, input_tas, sequence_length, t, prev_outputs, prev_state):
  """Performs a single RNN step optionally accounting for variable length."""
  outputs, state = core(
      tree.map_structure(lambda i: i.read(t), input_tas), prev_state)

  if prev_outputs is None:
    assert t == 0
    prev_outputs = tree.map_structure(tf.zeros_like, outputs)

  # TODO(slebedev): do not go into this block if t < min_len.
  if sequence_length is not None:
    # Selectively propagate outputs/state to the not-yet-finished
    # sequences.
    maybe_propagate = functools.partial(_safe_where, t >= sequence_length)
    outputs = tree.map_structure(maybe_propagate, prev_outputs, outputs)
    state = tree.map_structure(maybe_propagate, prev_state, state)

  return outputs, state


class VanillaRNN(RNNCore):
  """Basic fully-connected RNN core.

  Given :math:`x_t` and the previous hidden state :math:`h_{t-1}` the
  core computes

  .. math::

     h_t = w_i x_t + w_h h_{t-1} + b

  Attributes:
    input_to_hidden: Input-to-hidden weights :math:`w_i`, a tensor of shape
      ``[hidden_size, hidden_size]``.
    hidden_to_hidden: Hidden-to-hidden weights :math:`w_i`, a tensor of shape
      ``[input_size, hidden_size]``.
    b: bias, a tensor or shape ``[hidden_size]``.
  """

  def __init__(self,
               hidden_size: int,
               activation: types.ActivationFn = tf.tanh,
               w_i_init: Optional[initializers.Initializer] = None,
               w_h_init: Optional[initializers.Initializer] = None,
               b_init: Optional[initializers.Initializer] = None,
               dtype: tf.DType = tf.float32,
               name: Optional[str] = None):
    """Constructs a vanilla RNN core.

    Args:
      hidden_size: Hidden layer size.
      activation: Activation function to use. Defaults to ``tf.tanh``.
      w_i_init: Optional initializer for the input-to-hidden weights.
        Defaults to :class:`~initializers.TruncatedNormal` with a standard
        deviation of ``1 / sqrt(input_size)``.
      w_h_init: Optional initializer for the hidden-to-hidden weights.
        Defaults to :class:`~initializers.TruncatedNormal` with a standard
        deviation of ``1 / sqrt(hidden_size)``.
      b_init: Optional initializer for the bias. Defaults to
        :class:`~initializers.Zeros`.
      dtype: Optional :tf:`DType` of the core's variables. Defaults to
        ``tf.float32``.
      name: Name of the module.
    """
    super().__init__(name)
    self._hidden_size = hidden_size
    self._activation = activation
    self._b_init = b_init or initializers.Zeros()
    self._dtype = dtype

    self._input_to_hidden = linear.Linear(
        hidden_size, with_bias=False, w_init=w_i_init, name="input_to_hidden")
    self._hidden_to_hidden = linear.Linear(
        hidden_size, with_bias=False, w_init=w_h_init, name="hidden_to_hidden")

  @property
  def input_to_hidden(self) -> tf.Variable:
    return self._input_to_hidden.w

  @property
  def hidden_to_hidden(self) -> tf.Variable:
    return self._hidden_to_hidden.w

  def __call__(self, inputs: types.TensorNest,
               prev_state: types.TensorNest) -> Tuple[tf.Tensor, tf.Tensor]:
    """See base class."""
    self._initialize(inputs)

    outputs = self._activation(
        self._input_to_hidden(inputs) + self._hidden_to_hidden(prev_state) +
        self._b)

    # For VanillaRNN, the next state of the RNN is the same as the outputs.
    return outputs, outputs

  def initial_state(self, batch_size: int) -> tf.Tensor:
    """See base class."""
    return tf.zeros(shape=[batch_size, self._hidden_size], dtype=self._dtype)

  @once.once
  def _initialize(self, inputs: tf.Tensor):
    dtype = _check_inputs_dtype(inputs, self._dtype)
    self._b = tf.Variable(self._b_init([self._hidden_size], dtype), name="b")


class _LegacyDeepRNN(RNNCore):
  """Sonnet 1 compatible :class:`DeepRNN` implementation.

  This class is not intended to be used directly. Refer to :class:`DeepRNN`
  and ``deep_rnn_with_*_connections``.
  """

  def __init__(self,
               layers,
               skip_connections,
               concat_final_output_if_skip=True,
               name: Optional[str] = None):
    r"""Constructs a ``DeepRNN``.

    Args:
      layers: A list of :class:`RNNCore`\ s or callables.
      skip_connections: See :func:`deep_rnn_with_skip_connections`.
      concat_final_output_if_skip: See :func:`deep_rnn_with_skip_connections`.
      name: Name of the module.
    """
    super().__init__(name)
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
        current_inputs = tree.map_structure(concat, inputs, current_inputs)

      if isinstance(layer, RNNCore):
        current_inputs, next_state = layer(current_inputs,
                                           prev_state[recurrent_idx])
        next_states.append(next_state)
        recurrent_idx += 1
      else:
        current_inputs = layer(current_inputs)

      if self._skip_connections:
        outputs.append(current_inputs)

    if self._skip_connections and self._concat_final_output_if_skip:
      outputs = tree.map_structure(concat, *outputs)
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
  r"""Linear chain of :class:`RNNCore`\ s or callables.

  The core takes ``(input, prev_state)`` as input and passes the input
  through each internal module in the order they were presented, using
  elements from ``prev_state`` as necessary for internal RNN cores.

      >>> deep_rnn = snt.DeepRNN([
      ...     snt.LSTM(hidden_size=16),
      ...     snt.LSTM(hidden_size=16),
      ... ])

  Note that the state of a ``DeepRNN`` is always a tuple, which will
  contain the same number of elements as there are internal RNN cores.
  If no internal modules are RNN cores, the state of the ``DeepRNN`` as
  a whole is an empty tuple.

  Wrapping non-recurrent modules into a ``DeepRNN`` can be useful to
  produce something API compatible with a "real" recurrent module,
  simplifying code that handles the cores.
  """

  # TODO(slebedev): currently called `layers` to be in-sync with `Sequential`.
  def __init__(self, layers, name: Optional[str] = None):
    super().__init__(layers, skip_connections=False, name=name)


def deep_rnn_with_skip_connections(
    layers: Sequence[RNNCore],
    concat_final_output: bool = True,
    name: str = "deep_rnn_with_skip_connections") -> RNNCore:
  r"""Constructs a :class:`DeepRNN` with skip connections.

  Skip connections alter the dependency structure within a :class:`DeepRNN`.
  Specifically, input to the i-th layer (i > 0) is given by a
  concatenation of the core's inputs and the outputs of the (i-1)-th layer.
  ::

      outputs0, ... = layers[0](inputs, ...)
      outputs1, ... = layers[1](tf.concat([inputs, outputs0], axis=1], ...)
      outputs2, ... = layers[2](tf.concat([inputs, outputs1], axis=1], ...)
      ...

  This allows the layers to learn decoupled features.

  Args:
    layers: A list of :class:`RNNCore`\ s.
    concat_final_output: If enabled (default), the outputs of the core is a
      concatenation of the outputs of all intermediate layers; otherwise, only
      the outputs of the final layer, i.e. that of ``layers[-1]``, are returned.
    name: Name of the module.

  Returns:
    A :class:`DeepRNN` with skip connections.

  Raises:
    ValueError: If any of the layers is not an :class:`RNNCore`.
  """
  if not all(isinstance(l, RNNCore) for l in layers):
    raise ValueError("deep_rnn_with_skip_connections requires all layers to be "
                     "instances of RNNCore")

  return _LegacyDeepRNN(
      layers,
      skip_connections=True,
      concat_final_output_if_skip=concat_final_output,
      name=name)


class _ResidualWrapper(RNNCore):
  """Residual connection wrapper for a base :class:`RNNCore`.

  The output of the wrapper is the sum of the outputs of the base core
  with its inputs.
  """

  def __init__(self, base_core: RNNCore):
    super().__init__(name=base_core.name + "_residual")
    self._base_core = base_core

  def __call__(self, inputs: types.TensorNest, prev_state: types.TensorNest):
    """See base class."""
    outputs, next_state = self._base_core(inputs, prev_state)
    residual = tree.map_structure(lambda i, o: i + o, inputs, outputs)
    return residual, next_state

  def initial_state(self, batch_size, **kwargs):
    return self._base_core.initial_state(batch_size, **kwargs)


def deep_rnn_with_residual_connections(
    layers: Sequence[RNNCore],
    name: str = "deep_rnn_with_residual_connections") -> RNNCore:
  r"""Constructs a :class:`DeepRNN` with residual connections.

  Residual connections alter the dependency structure in a :class:`DeepRNN`.
  Specifically, the input to the i-th intermediate layer is a sum of
  the original core's inputs and the outputs of all the preceding
  layers (<i).
  ::

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
    layers: A list of :class:`RNNCore`\ s.
    name: Name of the module.

  Returns:
    A :class:`DeepRNN` with residual connections.

  Raises:
    ValueError: If any of the layers is not an :class:`RNNCore`.
  """
  if not all(isinstance(l, RNNCore) for l in layers):
    raise ValueError(
        "deep_rnn_with_residual_connections requires all layers to be "
        "instances of RNNCore")

  return _LegacyDeepRNN([_ResidualWrapper(l) for l in layers],
                        skip_connections=False,
                        name=name)


LSTMState = collections.namedtuple("LSTMState", ["hidden", "cell"])


class LSTM(RNNCore):
  r"""Long short-term memory (LSTM) RNN core.

  The implementation is based on :cite:`zaremba2014recurrent`. Given
  :math:`x_t` and the previous state :math:`(h_{t-1}, c_{t-1})` the core
  computes

  .. math::

     \begin{array}{ll}
     i_t = \sigma(W_{ii} x_t + W_{hi} h_{t-1} + b_i) \\
     f_t = \sigma(W_{if} x_t + W_{hf} h_{t-1} + b_f) \\
     g_t = \tanh(W_{ig} x_t + W_{hg} h_{t-1} + b_g) \\
     o_t = \sigma(W_{io} x_t + W_{ho} h_{t-1} + b_o) \\
     c_t = f_t c_{t-1} + i_t g_t \\
     h_t = o_t \tanh(c_t)
     \end{array}

  Where :math:`i_t`, :math:`f_t`, :math:`o_t` are input, forget and
  output gate activations, and :math:`g_t` is a vector of cell updates.

  Notes:
    Forget gate initialization:
      Following :cite:`jozefowicz2015empirical` we add a constant
      ``forget_bias`` (defaults to 1.0) to :math:`b_f` after initialization
      in order to reduce the scale of forgetting in the beginning of
      the training.
    Recurrent projections:
      Hidden state could be projected (via the ``project_size`` parameter)
      to reduce the number of parameters and speed up computation. For more
      details see :cite:`sak2014long`.

  Attributes:
    input_to_hidden: Input-to-hidden weights :math:`W_{ii}`, :math:`W_{if}`,
      :math:`W_{ig}` and :math:`W_{io}` concatenated into a tensor of shape
      ``[input_size, 4 * hidden_size]``.
    hidden_to_hidden: Hidden-to-hidden weights :math:`W_{hi}`, :math:`W_{hf}`,
      :math:`W_{hg}` and :math:`W_{ho}` concatenated into a tensor of shape
      ``[hidden_size, 4 * hidden_size]``.
    b: Biases :math:`b_i`, :math:`b_f`, :math:`b_g` and :math:`b_o` concatenated
      into a tensor of shape ``[4 * hidden_size]``.
  """

  def __init__(self,
               hidden_size: int,
               projection_size: Optional[int] = None,
               projection_init: Optional[initializers.Initializer] = None,
               w_i_init: Optional[initializers.Initializer] = None,
               w_h_init: Optional[initializers.Initializer] = None,
               b_init: Optional[initializers.Initializer] = None,
               forget_bias: types.FloatLike = 1.0,
               dtype: tf.DType = tf.float32,
               name: Optional[str] = None):
    """Constructs an LSTM.

    Args:
      hidden_size: Hidden layer size.
      projection_size: Optional int; if set, then the hidden state is projected
        to this size via a trainable projection matrix.
      projection_init: Optional initializer for the projection matrix.
        Defaults to :class:`~initializers.TruncatedNormal` with a standard
        deviation of ``1 / sqrt(hidden_size)``.
      w_i_init: Optional initializer for the input-to-hidden weights.
        Defaults to :class:`~initializers.TruncatedNormal` with a standard
        deviation of ``1 / sqrt(input_size)``.
      w_h_init: Optional initializer for the hidden-to-hidden weights.
        Defaults to :class:`~initializers.TruncatedNormal` with a standard
        deviation of ``1 / sqrt(hidden_size)``.
      b_init: Optional initializer for the biases. Defaults to
        :class:`~initializers.Zeros`.
      forget_bias: Optional float to add to the bias of the forget gate after
        initialization.
      dtype: Optional :tf:`DType` of the core's variables. Defaults to
        ``tf.float32``.
      name: Name of the module.
    """
    super().__init__(name)
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
    self._initialize(inputs)
    return _lstm_fn(inputs, prev_state, self._w_i, self._w_h, self.b,
                    self.projection)

  def initial_state(self, batch_size: int) -> LSTMState:
    """See base class."""
    return LSTMState(
        hidden=tf.zeros([batch_size, self._eff_hidden_size], dtype=self._dtype),
        cell=tf.zeros([batch_size, self._hidden_size], dtype=self._dtype))

  @property
  def input_to_hidden(self):
    return self._w_i

  @property
  def hidden_to_hidden(self):
    return self._w_h

  @once.once
  def _initialize(self, inputs):
    utils.assert_rank(inputs, 2)
    input_size = inputs.shape[1]
    dtype = _check_inputs_dtype(inputs, self._dtype)

    w_i_init = self._w_i_init or initializers.TruncatedNormal(
        stddev=1.0 / tf.sqrt(tf.cast(input_size, dtype)))
    w_h_init = self._w_h_init or initializers.TruncatedNormal(
        stddev=1.0 / tf.sqrt(tf.constant(self._eff_hidden_size, dtype=dtype)))
    self._w_i = tf.Variable(
        w_i_init([input_size, 4 * self._hidden_size], dtype), name="w_i")
    self._w_h = tf.Variable(
        w_h_init([self._eff_hidden_size, 4 * self._hidden_size], dtype),
        name="w_h")

    b_i, b_f, b_g, b_o = tf.split(
        self._b_init([4 * self._hidden_size], dtype), num_or_size_splits=4)
    b_f += self._forget_bias
    self.b = tf.Variable(tf.concat([b_i, b_f, b_g, b_o], axis=0), name="b")

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


def _lstm_fn(inputs, prev_state, w_i, w_h, b, projection=None):
  """Compute one step of an LSTM."""
  gates_x = tf.matmul(inputs, w_i)
  gates_h = tf.matmul(prev_state.hidden, w_h)
  gates = gates_x + gates_h + b

  # i = input, f = forget, g = cell updates, o = output.
  i, f, g, o = tf.split(gates, num_or_size_splits=4, axis=1)

  next_cell = tf.sigmoid(f) * prev_state.cell
  next_cell += tf.sigmoid(i) * tf.tanh(g)
  next_hidden = tf.sigmoid(o) * tf.tanh(next_cell)

  if projection is not None:
    next_hidden = tf.matmul(next_hidden, projection)

  return next_hidden, LSTMState(hidden=next_hidden, cell=next_cell)


class UnrolledLSTM(UnrolledRNN):
  """Unrolled long short-term memory (LSTM).

  The implementation uses efficient device-specialized ops, e.g. CuDNN-RNN
  on a CUDA-enabled GPU, and can be an order of magnitude faster than
  ``snt.*_unroll`` with an :class:`LSTM` core.
  """

  def __init__(self,
               hidden_size,
               w_i_init: Optional[initializers.Initializer] = None,
               w_h_init: Optional[initializers.Initializer] = None,
               b_init: Optional[initializers.Initializer] = None,
               forget_bias: types.FloatLike = 1.0,
               dtype: tf.DType = tf.float32,
               name: Optional[str] = None):
    """Construct an unrolled LSTM.

    Args:
      hidden_size: Hidden layer size.
      w_i_init: Optional initializer for the input-to-hidden weights.
        Defaults to :class:`~initializers.TruncatedNormal` with a standard
        deviation of ``1 / sqrt(input_size)``.
      w_h_init: Optional initializer for the hidden-to-hidden weights.
        Defaults to :class:`~initializers.TruncatedNormal` with a standard
        deviation of ``1 / sqrt(hidden_size)``.
      b_init: Optional initializer for the biases. Defaults to
        :class:`~initializers.Zeros`.
      forget_bias: Optional float to add to the bias of the forget gate after
        initialization.
      dtype: Optional :tf:`DType` of the core's variables. Defaults to
        ``tf.float32``.
      name: Name of the module.
    """
    super().__init__(name)
    self._hidden_size = hidden_size
    self._w_i_init = w_i_init
    self._w_h_init = w_h_init
    self._b_init = b_init or initializers.Zeros()
    self._forget_bias = forget_bias
    self._dtype = dtype

  def __call__(self, input_sequence, initial_state):
    """See base class."""
    self._initialize(input_sequence)
    return _specialized_unrolled_lstm(input_sequence, initial_state, self._w_i,
                                      self._w_h, self.b)

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
  def _initialize(self, input_sequence):
    utils.assert_rank(input_sequence, 3)  # [num_steps, batch_size, input_size].
    input_size = input_sequence.shape[2]
    dtype = _check_inputs_dtype(input_sequence, self._dtype)

    w_i_init = self._w_i_init or initializers.TruncatedNormal(
        stddev=1.0 / tf.sqrt(tf.cast(input_size, dtype)))
    w_h_init = self._w_h_init or initializers.TruncatedNormal(
        stddev=1.0 / tf.sqrt(tf.constant(self._hidden_size, dtype=dtype)))
    self._w_i = tf.Variable(
        w_i_init([input_size, 4 * self._hidden_size], dtype), name="w_i")
    self._w_h = tf.Variable(
        w_h_init([self._hidden_size, 4 * self._hidden_size], dtype), name="w_h")

    b_i, b_f, b_g, b_o = tf.split(
        self._b_init([4 * self._hidden_size], dtype), num_or_size_splits=4)
    b_f += self._forget_bias
    self.b = tf.Variable(tf.concat([b_i, b_f, b_g, b_o], axis=0), name="b")


# TODO(b/133740216): consider upstreaming into TensorFlow.
def _specialize_per_device(api_name, specializations, default):
  """Create a :tf:`function` specialized per-device.

  Args:
    api_name: Name of the function, e.g. ``"lstm"``.
    specializations: A mapping from device type (e.g. ``"CPU"`` or ``"TPU``) to
      a Python function with a specialized implementation for that device.
    default: Default device type to use (typically, ``"CPU"``).

  Returns:
    A :tf:`function` which when called dispatches to the specialization
    for the current device.
  """
  # Cached to avoid redundant ``ModuleWrapper.__getattribute__`` calls.
  list_logical_devices = tf.config.experimental.list_logical_devices

  def wrapper(*args, **kwargs):
    """Specialized {}.

    In eager mode the specialization is chosen based on the current
    device context or, if no device context is active, on availability
    of a GPU.

    In graph mode (inside tf.function) the choice is delegated to the
    implementation selector pass in Grappler.

    Args:
      *args: Positional arguments to pass to the chosen specialization.
      **kwargs: Keyword arguments to pass to the chosen specialization.
    """.format(api_name)
    ctx = context_lib.context()
    if ctx.executing_eagerly():
      device = ctx.device_spec.device_type
      if device is None:
        # Soft-placement will never implicitly place an op an a TPU, so
        # we only need to consider CPU/GPU.
        device = "GPU" if list_logical_devices("GPU") else "CPU"

      specialization = specializations.get(device) or specializations[default]
      return specialization(*args, **kwargs)

    # Implementation selector requires a globally unique name for each
    # .register() call.
    unique_api_name = "{}_{}".format(api_name, uuid.uuid4())
    functions = {}
    for device, specialization in specializations.items():
      functions[device] = function_lib.defun_with_attributes(
          specialization,
          attributes={
              "api_implements": unique_api_name,
              "api_preferred_device": device
          })
      concrete_func = functions[device].get_concrete_function(*args, **kwargs)
      concrete_func.add_to_graph()
      concrete_func.add_gradient_functions_to_graph()
    return functions[default](*args, **kwargs)

  return wrapper


def _fallback_unrolled_lstm(input_sequence, initial_state, w_i, w_h, b):
  """Fallback version of :class:`UnrolledLSTM` which works on any device."""
  return dynamic_unroll(
      functools.partial(_lstm_fn, w_i=w_i, w_h=w_h, b=b), input_sequence,
      initial_state)


def _block_unrolled_lstm(input_sequence, initial_state, w_i, w_h, b):
  """Efficient CPU specialization of :class:`UnrolledLSTM`."""
  w_peephole = tf.zeros(
      tf.shape(initial_state.hidden)[1:], dtype=initial_state.hidden.dtype)
  _, all_cell, _, _, _, _, all_hidden = tf.raw_ops.BlockLSTMV2(
      seq_len_max=tf.cast(tf.shape(input_sequence)[0], tf.int64),
      x=input_sequence,
      cs_prev=initial_state.cell,
      h_prev=initial_state.hidden,
      w=tf.concat([w_i, w_h], axis=0),
      wci=w_peephole,
      wcf=w_peephole,
      wco=w_peephole,
      b=b,
      use_peephole=False)
  return all_hidden, LSTMState(all_hidden[-1], all_cell[-1])


def _cudnn_unrolled_lstm(input_sequence, initial_state, w_i, w_h, b):
  """GPU/CuDNN-RNN specialization of :class:`UnrolledLSTM`."""
  # Intuitively, concat/transpose is not free but we did not see
  # it significantly affecting performance in benchmarks.
  output_sequence, all_hidden, all_cell, _ = tf.raw_ops.CudnnRNN(
      input=input_sequence,
      input_h=tf.expand_dims(initial_state.hidden, axis=0),
      input_c=tf.expand_dims(initial_state.cell, axis=0),
      params=tf.concat(
          [
              tf.reshape(tf.transpose(w_i), [-1]),
              tf.reshape(tf.transpose(w_h), [-1]),
              b,
              # CuDNN has two sets of biases: b_i and b_h, zero-out b_h.
              tf.zeros_like(b),
          ],
          axis=0),
      rnn_mode="lstm")
  return output_sequence, LSTMState(all_hidden[-1], all_cell[-1])


_unrolled_lstm_impls = {
    "GPU": _cudnn_unrolled_lstm,
    "TPU": _fallback_unrolled_lstm,
}
# TODO(tomhennigan) Remove this check when TF 2.1 is released.
if hasattr(tf.raw_ops, "BlockLSTMV2"):
  _unrolled_lstm_impls["CPU"] = _block_unrolled_lstm

_specialized_unrolled_lstm = _specialize_per_device(
    "snt_unrolled_lstm", specializations=_unrolled_lstm_impls, default="TPU")


class _RecurrentDropoutWrapper(RNNCore):
  """Recurrent dropout wrapper for a base RNN core.

  The wrapper drops the previous state of the base core according to
  dropout ``rates``. Specifically, dropout is only applied if the rate
  corresponding to the state element is not `None`. Dropout masks
  are sampled in `initial_state` of the wrapper.

  This class is not intended to be used directly. See
  ``lstm_with_recurrent_dropout``.
  """

  def __init__(self, base_core: RNNCore, rates, seed: Optional[int] = None):
    """Wraps a given base RNN core.

    Args:
      base_core: The ``RNNCore`` to be wrapped
      rates: Recurrent dropout probabilities. The structure should match that of
        ``base_core.initial_state``.
      seed: Optional int; seed passed to :tf:`nn.dropout`.
    """
    super().__init__(name=base_core.name + "_recurrent_dropout")
    self._base_core = base_core
    self._rates = rates
    self._seed = seed

  def __call__(self, inputs, prev_state):
    prev_core_state, dropout_masks = prev_state
    prev_core_state = tree.map_structure(
        lambda s, mask: s  # pylint: disable=g-long-lambda
        if mask is None else s * mask,
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

    dropout_masks = tree.map_structure(maybe_dropout, core_initial_state,
                                       self._rates)
    return core_initial_state, dropout_masks


def lstm_with_recurrent_dropout(hidden_size, dropout=0.5, seed=None, **kwargs):
  r"""Constructs an LSTM with recurrent dropout.

  The implementation is based on :cite:`gal2016theoretically`. Dropout
  is applied on the previous hidden state :math:`h_{t-1}` during the
  computation of gate activations:

  .. math::

     \begin{array}{ll}
     i_t = \sigma(W_{ii} x_t + W_{hi} d(h_{t-1}) + b_i) \\
     f_t = \sigma(W_{if} x_t + W_{hf} d(h_{t-1}) + b_f) \\
     g_t = \tanh(W_{ig} x_t + W_{hg} d(h_{t-1}) + b_g) \\
     o_t = \sigma(W_{io} x_t + W_{ho} d(h_{t-1}) + b_o)
     \end{array}

  Args:
    hidden_size: Hidden layer size.
    dropout: Dropout probability.
    seed: Optional int; seed passed to :tf:`nn.dropout`.
    **kwargs: Optional keyword arguments to pass to the :class:`LSTM`
      constructor.

  Returns:
    A tuple of two elements:
      * **train_lstm** - An :class:`LSTM` with recurrent dropout enabled for
        training.
      * **test_lstm** - The same as ``train_lstm`` but without recurrent
        dropout.

  Raises:
    ValueError: If ``dropout`` is not in ``[0, 1)``.
  """
  if dropout < 0 or dropout >= 1:
    raise ValueError(
        "dropout must be in the range [0, 1), got {}".format(dropout))

  lstm = LSTM(hidden_size, **kwargs)
  rate = LSTMState(hidden=dropout, cell=None)
  return _RecurrentDropoutWrapper(lstm, rate, seed), lstm


class _ConvNDLSTM(RNNCore):
  r"""``num_spatial_dims``-D convolutional LSTM.

  The implementation is based on :cite:`xingjian2015convolutional`.
  Given :math:`x_t` and the previous state :math:`(h_{t-1}, c_{t-1})`
  the core computes

  .. math::

     \begin{array}{ll}
     i_t = \sigma(W_{ii} * x_t + W_{hi} * h_{t-1} + b_i) \\
     f_t = \sigma(W_{if} * x_t + W_{hf} * h_{t-1} + b_f) \\
     g_t = \tanh(W_{ig} * x_t + W_{hg} * h_{t-1} + b_g) \\
     o_t = \sigma(W_{io} * x_t + W_{ho} * h_{t-1} + b_o) \\
     c_t = f_t c_{t-1} + i_t g_t \\
     h_t = o_t \tanh(c_t)
     \end{array}

  where :math:`*` denotes the convolution operator; :math:`i_t`,
  :math:`f_t`, :math:`o_t` are input, forget and output gate activations,
  and :math:`g_t` is a vector of cell updates.

  Notes:
    Forget gate initialization:
      Following :cite:`jozefowicz2015empirical` we add a constant
      ``forget_bias`` (defaults to 1.0) to :math:`b_f` after initialization
      in order to reduce the scale of forgetting in the beginning of
      the training.

  Attributes:
    input_to_hidden: Input-to-hidden convolution weights :math:`W_{ii}`,
      :math:`W_{if}`, :math:`W_{ig}` and :math:`W_{io}` concatenated into a
      single tensor of shape ``[kernel_shape*, input_channels, 4 *
      output_channels]`` where ``kernel_shape`` is repeated ``num_spatial_dims``
      times.
    hidden_to_hidden: Hidden-to-hidden convolution weights :math:`W_{hi}`,
      :math:`W_{hf}`, :math:`W_{hg}` and :math:`W_{ho}` concatenated into a
      single tensor of shape ``[kernel_shape*, input_channels, 4 *
      output_channels]`` where ``kernel_shape`` is repeated ``num_spatial_dims``
      times.
    b: Biases :math:`b_i`, :math:`b_f`, :math:`b_g` and :math:`b_o` concatenated
      into a tensor of shape ``[4 * output_channels]``.
  """

  def __init__(self,
               num_spatial_dims: int,
               input_shape: types.ShapeLike,
               output_channels: int,
               kernel_shape: Union[int, Sequence[int]],
               data_format: Optional[str] = None,
               w_i_init: Optional[initializers.Initializer] = None,
               w_h_init: Optional[initializers.Initializer] = None,
               b_init: Optional[initializers.Initializer] = None,
               forget_bias: types.FloatLike = 1.0,
               dtype: tf.DType = tf.float32,
               name: Optional[str] = None):
    """Constructs a convolutional LSTM.

    Args:
      num_spatial_dims: Number of spatial dimensions of the input.
      input_shape: Shape of the inputs excluding batch size.
      output_channels: Number of output channels.
      kernel_shape: Sequence of kernel sizes (of length ``num_spatial_dims``),
        or an int. ``kernel_shape`` will be expanded to define a kernel size in
        all dimensions.
      data_format: The data format of the input.
      w_i_init: Optional initializer for the input-to-hidden convolution
        weights. Defaults to :class:`~initializers.TruncatedNormal` with a
        standard deviation of ``1 / sqrt(kernel_shape**num_spatial_dims *
        input_channels)``.
      w_h_init: Optional initializer for the hidden-to-hidden convolution
        weights. Defaults to :class:`~initializers.TruncatedNormal` with a
        standard deviation of ``1 / sqrt(kernel_shape**num_spatial_dims *
        input_channels)``.
      b_init: Optional initializer for the biases. Defaults to
        :class:`~initializers.Zeros`.
      forget_bias: Optional float to add to the bias of the forget gate after
        initialization.
      dtype: Optional :tf:`DType` of the core's variables. Defaults to
        ``tf.float32``.
      name: Name of the module.
    """
    super().__init__(name)
    self._num_spatial_dims = num_spatial_dims
    self._input_shape = list(input_shape)
    self._channel_index = 1 if (data_format is not None and
                                data_format.startswith("NC")) else -1
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
    self._initialize(inputs)

    gates = self._input_to_hidden(inputs)
    gates += self._hidden_to_hidden(prev_state.hidden)
    gates += self.b

    # i = input, f = forget, g = cell updates, o = output.
    i, f, g, o = tf.split(
        gates, num_or_size_splits=4, axis=self._num_spatial_dims + 1)

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
  def _initialize(self, inputs):
    dtype = _check_inputs_dtype(inputs, self._dtype)
    b_i, b_f, b_g, b_o = tf.split(
        self._b_init([4 * self._output_channels], dtype), num_or_size_splits=4)
    b_f += self._forget_bias
    self.b = tf.Variable(tf.concat([b_i, b_f, b_g, b_o], axis=0), name="b")


class Conv1DLSTM(_ConvNDLSTM):  # pylint: disable=missing-docstring,empty-docstring
  __doc__ = _ConvNDLSTM.__doc__.replace("``num_spatial_dims``", "1")

  def __init__(self,
               input_shape: types.ShapeLike,
               output_channels: int,
               kernel_shape: Union[int, Sequence[int]],
               data_format="NWC",
               w_i_init: Optional[initializers.Initializer] = None,
               w_h_init: Optional[initializers.Initializer] = None,
               b_init: Optional[initializers.Initializer] = None,
               forget_bias: types.FloatLike = 1.0,
               dtype: tf.DType = tf.float32,
               name: Optional[str] = None):
    """Constructs a 1-D convolutional LSTM.

    Args:
      input_shape: Shape of the inputs excluding batch size.
      output_channels: Number of output channels.
      kernel_shape: Sequence of kernel sizes (of length 1), or an int.
        ``kernel_shape`` will be expanded to define a kernel size in all
        dimensions.
      data_format: The data format of the input.
      w_i_init: Optional initializer for the input-to-hidden convolution
        weights. Defaults to :class:`~initializers.TruncatedNormal` with a
        standard deviation of ``1 / sqrt(kernel_shape * input_channels)``.
      w_h_init: Optional initializer for the hidden-to-hidden convolution
        weights. Defaults to :class:`~initializers.TruncatedNormal` with a
        standard deviation of ``1 / sqrt(kernel_shape * input_channels)``.
      b_init: Optional initializer for the biases. Defaults to
        :class:`~initializers.Zeros`.
      forget_bias: Optional float to add to the bias of the forget gate after
        initialization.
      dtype: Optional :tf:`DType` of the core's variables. Defaults to
        ``tf.float32``.
      name: Name of the module.
    """
    super().__init__(
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


class Conv2DLSTM(_ConvNDLSTM):  # pylint: disable=missing-docstring,empty-docstring
  __doc__ = _ConvNDLSTM.__doc__.replace("``num_spatial_dims``", "2")

  def __init__(self,
               input_shape: types.ShapeLike,
               output_channels: int,
               kernel_shape: Union[int, Sequence[int]],
               data_format: str = "NHWC",
               w_i_init: Optional[initializers.Initializer] = None,
               w_h_init: Optional[initializers.Initializer] = None,
               b_init: Optional[initializers.Initializer] = None,
               forget_bias: types.FloatLike = 1.0,
               dtype: tf.DType = tf.float32,
               name: Optional[str] = None):
    """Constructs a 2-D convolutional LSTM.

    Args:
      input_shape: Shape of the inputs excluding batch size.
      output_channels: Number of output channels.
      kernel_shape: Sequence of kernel sizes (of length 2), or an int.
        ``kernel_shape`` will be expanded to define a kernel size in all
        dimensions.
      data_format: The data format of the input.
      w_i_init: Optional initializer for the input-to-hidden convolution
        weights. Defaults to :class:`~initializers.TruncatedNormal` with a
        standard deviation of ``1 / sqrt(kernel_shape**2 * input_channels)``.
      w_h_init: Optional initializer for the hidden-to-hidden convolution
        weights. Defaults to :class:`~initializers.TruncatedNormal` with a
        standard deviation of ``1 / sqrt(kernel_shape**2 * input_channels)``.
      b_init: Optional initializer for the biases. Defaults to
        :class:`~initializers.Zeros`.
      forget_bias: Optional float to add to the bias of the forget gate after
        initialization.
      dtype: Optional :tf:`DType` of the core's variables. Defaults to
        ``tf.float32``.
      name: Name of the module.
    """
    super().__init__(
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


class Conv3DLSTM(_ConvNDLSTM):  # pylint: disable=missing-docstring,empty-docstring
  __doc__ = _ConvNDLSTM.__doc__.replace("``num_spatial_dims``", "3")

  def __init__(self,
               input_shape: types.ShapeLike,
               output_channels: int,
               kernel_shape: Union[int, Sequence[int]],
               data_format: str = "NDHWC",
               w_i_init: Optional[initializers.Initializer] = None,
               w_h_init: Optional[initializers.Initializer] = None,
               b_init: Optional[initializers.Initializer] = None,
               forget_bias: types.FloatLike = 1.0,
               dtype: tf.DType = tf.float32,
               name: Optional[str] = None):
    """Constructs a 3-D convolutional LSTM.

    Args:
      input_shape: Shape of the inputs excluding batch size.
      output_channels: Number of output channels.
      kernel_shape: Sequence of kernel sizes (of length 3), or an int.
        ``kernel_shape`` will be expanded to define a kernel size in all
        dimensions.
      data_format: The data format of the input.
      w_i_init: Optional initializer for the input-to-hidden convolution
        weights. Defaults to :class:`~initializers.TruncatedNormal` with a
        standard deviation of ``1 / sqrt(kernel_shape**3 * input_channels)``.
      w_h_init: Optional initializer for the hidden-to-hidden convolution
        weights. Defaults to :class:`~initializers.TruncatedNormal` with a
        standard deviation of ``1 / sqrt(kernel_shape**3 * input_channels)``.
      b_init: Optional initializer for the biases. Defaults to
        :class:`~initializers.Zeros`.
      forget_bias: Optional float to add to the bias of the forget gate after
        initialization.
      dtype: Optional :tf:`DType` of the core's variables. Defaults to
        ``tf.float32``.
      name: Name of the module.
    """
    super().__init__(
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
  r"""Gated recurrent unit (GRU) RNN core.

  The implementation is based on :cite:`chung2014empirical`. Given
  :math:`x_t` and the previous state :math:`h_{t-1}` the core computes

  .. math::

     \begin{array}{ll}
     z_t &= \sigma(W_{iz} x_t + W_{hz} h_{t-1} + b_z) \\
     r_t &= \sigma(W_{ir} x_t + W_{hr} h_{t-1} + b_r) \\
     a_t &= \tanh(W_{ia} x_t + W_{ha} (r_t h_{t-1}) + b_a) \\
     h_t &= (1 - z_t) h_{t-1} + z_t a_t
     \end{array}

  where :math:`z_t` and :math:`r_t` are reset and update gates.

  Attributes:
    input_to_hidden: Input-to-hidden weights :math:`W_{iz}`, :math:`W_{ir}`
      and :math:`W_{ia}` concatenated into a tensor of shape
      ``[input_size, 3 * hidden_size]``.
    hidden_to_hidden: Hidden-to-hidden weights :math:`W_{hz}`, :math:`W_{hr}`
      and :math:`W_{ha}` concatenated into a tensor of shape
      ``[hidden_size, 3 * hidden_size]``.
    b: Biases :math:`b_z`, :math:`b_r` and :math:`b_a` concatenated into a
      tensor of shape ``[3 * hidden_size]``.
  """

  def __init__(self,
               hidden_size,
               w_i_init: Optional[initializers.Initializer] = None,
               w_h_init: Optional[initializers.Initializer] = None,
               b_init: Optional[initializers.Initializer] = None,
               dtype: tf.DType = tf.float32,
               name: Optional[str] = None):
    """Constructs a GRU.

    Args:
      hidden_size: Hidden layer size.
      w_i_init: Optional initializer for the input-to-hidden weights. Defaults
        to Glorot uniform initializer.
      w_h_init: Optional initializer for the hidden-to-hidden weights. Defaults
        to Glorot uniform initializer.
      b_init: Optional initializer for the biases. Defaults to
        :class:`~initializers.Zeros`.
      dtype: Optional :tf:`DType` of the core's variables. Defaults to
        ``tf.float32``.
      name: Name of the module.
    """
    super().__init__(name)
    self._hidden_size = hidden_size
    glorot_uniform = initializers.VarianceScaling(
        mode="fan_avg", distribution="uniform")
    self._w_i_init = w_i_init or glorot_uniform
    self._w_h_init = w_h_init or glorot_uniform
    self._b_init = b_init or initializers.Zeros()
    self._dtype = dtype

  def __call__(self, inputs, prev_state):
    """See base class."""
    self._initialize(inputs)

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
  def _initialize(self, inputs):
    utils.assert_rank(inputs, 2)
    input_size = inputs.shape[1]
    dtype = _check_inputs_dtype(inputs, self._dtype)
    self._w_i = tf.Variable(
        self._w_i_init([input_size, 3 * self._hidden_size], dtype), name="w_i")
    self._w_h = tf.Variable(
        self._w_h_init([self._hidden_size, 3 * self._hidden_size], dtype),
        name="w_h")
    self.b = tf.Variable(self._b_init([3 * self._hidden_size], dtype), name="b")


# TODO(slebedev): remove or document and export.
class CuDNNGRU(RNNCore):
  """Gated recurrent unit (GRU) RNN core implemented using CuDNN-RNN.

  The (CuDNN) implementation is based on https://arxiv.org/abs/1406.1078
  and differs from `GRU` in the way a_t and h_t are computed:

      a_t = tanh(W_{ia} x_t + r_t (W_{ha} h_{t-1}) + b_a)
      h_t = (1 - z_t) a_t + z_t h_{t-1}

  Unlike `GRU` this core operates on the whole batch of sequences at
  once, i.e. the expected shape of `inputs` is
  `[num_steps, batch_size, input_size]`.
  """

  def __init__(self,
               hidden_size,
               w_i_init: Optional[initializers.Initializer] = None,
               w_h_init: Optional[initializers.Initializer] = None,
               b_init: Optional[initializers.Initializer] = None,
               dtype: tf.DType = tf.float32,
               name: Optional[str] = None):
    """Constructs a `GRU`.

    Args:
      hidden_size: Hidden layer size.
      w_i_init: Optional initializer for the input-to-hidden weights. Defaults
        to Glorot uniform initializer.
      w_h_init: Optional initializer for the hidden-to-hidden weights. Defaults
        to Glorot uniform initializer.
      b_init: Optional initializer for the biases. Defaults to
        :class:`~initializers.Zeros`.
      dtype: Optional :tf:`DType` of the core's variables. Defaults to
        ``tf.float32``.
      name: Name of the module.
    """
    super().__init__(name)
    self._hidden_size = hidden_size
    glorot_uniform = initializers.VarianceScaling(
        mode="fan_avg", distribution="uniform")
    self._w_i_init = w_i_init or glorot_uniform
    self._w_h_init = w_h_init or glorot_uniform
    self._b_init = b_init or initializers.Zeros()
    self._dtype = dtype

  def __call__(self, inputs, prev_state):
    """See base class."""
    self._initialize(inputs)

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
        params=tf.concat(
            [
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
            ],
            axis=0),
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
  def _initialize(self, inputs):
    utils.assert_rank(inputs, 3)  # [num_steps, batch_size, input_size].
    input_size = inputs.shape[2]
    dtype = _check_inputs_dtype(inputs, self._dtype)
    self._w_i = tf.Variable(
        self._w_i_init([input_size, 3 * self._hidden_size], dtype), name="w_i")
    self._w_h = tf.Variable(
        self._w_h_init([self._hidden_size, 3 * self._hidden_size], dtype),
        name="w_h")
    self.b = tf.Variable(self._b_init([3 * self._hidden_size], dtype), name="b")


def _check_inputs_dtype(inputs, expected_dtype):
  if inputs.dtype is not expected_dtype:
    raise TypeError("inputs must have dtype {!r}, got {!r}".format(
        expected_dtype, inputs.dtype))
  return expected_dtype
