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

"""Basic RNN Cores for TensorFlow snt.

This file contains the definitions of the simplest building blocks for Recurrent
Neural Networks.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

# Dependency imports
import six

from sonnet.python.modules import base
from sonnet.python.modules import basic
from sonnet.python.modules import rnn_core
from sonnet.python.modules import util
import tensorflow as tf

from tensorflow.python.framework import tensor_shape
from tensorflow.python.util import nest


def _get_flat_core_sizes(cores):
  """Obtains the list flattened output sizes of a list of cores.

  Args:
    cores: list of cores to get the shapes from.

  Returns:
    List of lists that, for each core, contains the list of its output
      dimensions.
  """
  core_sizes_lists = []
  for core in cores:
    flat_output_size = nest.flatten(core.output_size)
    core_sizes_lists.append([tensor_shape.as_shape(size).as_list()
                             for size in flat_output_size])
  return core_sizes_lists


def _get_shape_without_batch_dimension(tensor_nest):
  """Converts Tensor nest to a TensorShape nest, removing batch dimension."""
  def _strip_batch_and_convert_to_shape(tensor):
    return tensor.get_shape()[1:]
  return nest.map_structure(_strip_batch_and_convert_to_shape, tensor_nest)


class VanillaRNN(rnn_core.RNNCore):
  """Basic fully connected vanilla RNN core."""

  IN_TO_HIDDEN = "in_to_hidden"
  HIDDEN_TO_HIDDEN = "hidden_to_hidden"
  POSSIBLE_INITIALIZER_KEYS = {IN_TO_HIDDEN, HIDDEN_TO_HIDDEN}

  def __init__(self, hidden_size, activation=tf.tanh, initializers=None,
               partitioners=None, regularizers=None, name="vanilla_rnn"):
    """Construct a Basic RNN core.

    Args:
      hidden_size: hidden size dimensionality.
      activation: activation function to use.
      initializers: optional dict containing ops to initialize the weights. This
        dictionary may contain the keys 'in_to_hidden' and/or
        'hidden_to_hidden'.
      partitioners: optional dict containing ops to partition the weights. This
        dictionary may contain the keys 'in_to_hidden' and/or
        'hidden_to_hidden'.
      regularizers: optional dict containing ops to regularize the weights. This
        dictionary may contain the keys 'in_to_hidden' and/or
        'hidden_to_hidden'.
      name: name of the module.

    Raises:
      KeyError: if `initializers` contains any keys other than 'in_to_hidden' or
        'hidden_to_hidden'.
      KeyError: if `partitioners` contains any keys other than 'in_to_hidden' or
        'hidden_to_hidden'.
      KeyError: if `regularizers` contains any keys other than 'in_to_hidden' or
        'hidden_to_hidden'.
      TypeError: If any of the given initializers are not callable.
      TypeError: If any of the given partitioners are not callable.
      TypeError: If any of the given regularizers are not callable.
    """
    super(VanillaRNN, self).__init__(name=name)
    self._hidden_size = hidden_size
    self._activation = activation
    self._initializers = util.check_initializers(
        initializers, self.POSSIBLE_INITIALIZER_KEYS)
    self._partitioners = util.check_partitioners(
        partitioners, self.POSSIBLE_INITIALIZER_KEYS)
    self._regularizers = util.check_regularizers(
        regularizers, self.POSSIBLE_INITIALIZER_KEYS)

  def _build(self, input_, prev_state):
    """Connects the VanillaRNN module into the graph.

    If this is not the first time the module has been connected to the graph,
    the Tensors provided as input_ and state must have the same final
    dimension, in order for the existing variables to be the correct size for
    their corresponding multiplications. The batch size may differ for each
    connection.

    Args:
      input_: a 2D Tensor of size [batch_size, input_size].
      prev_state: a 2D Tensor of size [batch_size, hidden_size].

    Returns:
      output: a 2D Tensor of size [batch_size, hidden_size].
      next_state: a Tensor of size [batch_size, hidden_size].

    Raises:
      ValueError: if connecting the module into the graph any time after the
        first time, and the inferred size of the inputs does not match previous
        invocations.
    """
    self._in_to_hidden_linear = basic.Linear(
        self._hidden_size, name="in_to_hidden",
        initializers=self._initializers.get("in_to_hidden"),
        partitioners=self._partitioners.get("in_to_hidden"),
        regularizers=self._regularizers.get("in_to_hidden"))

    self._hidden_to_hidden_linear = basic.Linear(
        self._hidden_size, name="hidden_to_hidden",
        initializers=self._initializers.get("hidden_to_hidden"),
        partitioners=self._partitioners.get("hidden_to_hidden"),
        regularizers=self._regularizers.get("hidden_to_hidden"))

    in_to_hidden = self._in_to_hidden_linear(input_)
    hidden_to_hidden = self._hidden_to_hidden_linear(prev_state)
    output = self._activation(in_to_hidden + hidden_to_hidden)

    # For VanillaRNN, the next state of the RNN is the same as the output
    return output, output

  @property
  def in_to_hidden_linear(self):
    self._ensure_is_connected()
    return self._in_to_hidden_linear

  @property
  def hidden_to_hidden_linear(self):
    self._ensure_is_connected()
    return self._hidden_to_hidden_linear

  @property
  def in_to_hidden_variables(self):
    self._ensure_is_connected()
    return self._in_to_hidden_linear.get_variables()

  @property
  def hidden_to_hidden_variables(self):
    self._ensure_is_connected()
    return self._hidden_to_hidden_linear.get_variables()

  @property
  def state_size(self):
    return tf.TensorShape([self._hidden_size])

  @property
  def output_size(self):
    return tf.TensorShape([self._hidden_size])


class DeepRNN(rnn_core.RNNCore):
  """RNN core that passes data through a number of internal modules or ops.

  This module is constructed by passing an iterable of externally constructed
  modules or ops. The DeepRNN takes `(input, prev_state)` as input and passes
  the input through each internal module in the order they were presented,
  using elements from `prev_state` as necessary for internal recurrent cores.
  The output is `(output, next_state)` in common with other RNN cores.
  By default, skip connections from the input to all internal modules and from
  each intermediate output to the final output are used.

  E.g.:

  ```python
  lstm1 = snt.LSTM(hidden_size=256)
  lstm2 = snt.LSTM(hidden_size=256)
  deep_rnn = snt.DeepRNN([lstm1, lstm2])
  output, next_state = deep_rnn(input, prev_state)
  ```

  The computation set up inside the DeepRNN has the same effect as:

  ```python
  prev_state1, prev_state2 = prev_state
  lstm1_output, next_state1 = lstm1(input, prev_state1)
  lstm2_output, next_state2 = lstm2(
      tf.concat([input, lstm1_output], 1), prev_state2)

  next_state = (next_state1, next_state2)
  output = tf.concat([lstm1_output, lstm2_output], 1)
  ```

  Every internal module receives the preceding module's output and the entire
  core's input. The output is created by concatenating each internal module's
  output. In the case of internal recurrent elements, corresponding elements
  of the state are used such that `state[i]` is passed to the `i`'th internal
  recurrent element. Note that the state of a `DeepRNN` is always a tuple, which
  will contain the same number of elements as there are internal recurrent
  cores. If no internal modules are recurrent, the state of the DeepRNN as a
  whole is the empty tuple. Wrapping non-recurrent modules into a DeepRNN can
  be useful to produce something API compatible with a "real" recurrent module,
  simplifying code that handles the cores.

  Without skip connections the previous example would become the following
  (note the only difference is the addition of `skip_connections=False`):

  ```python
  # ... declare other modules as above
  deep_rnn = snt.DeepRNN([lin, tanh, lstm], skip_connections=False)
  output, next_state = deep_rnn(input, prev_state)
  ```

  which is equivalent to:

  ```python
  lin_output = lin(input)
  tanh_output = tanh(lin_output)
  lstm_output, lstm_next_state = lstm(tanh_output, prev_state[0])

  next_state = (lstm_next_state,)
  output = lstm_output
  ```

  Note: when using skip connections, all the cores should be recurrent.
  """

  def __init__(self, cores, skip_connections=True,
               concat_final_output_if_skip=True, name="deep_rnn"):
    """Construct a Deep RNN core.

    Args:
      cores: iterable of modules or ops.
      skip_connections: a boolean that indicates whether to use skip
        connections. This means that the input is fed to all the layers, after
        being concatenated on the last dimension with the output of the previous
        layer. The output of the module will be the concatenation of all the
        outputs of the internal modules.
      concat_final_output_if_skip: A boolean that indicates whether the outputs
        of intermediate layers should be concatenated into the timestep-wise
        output of the core. By default this is True. If this is set to False,
        then the core output is that of the final layer, i.e. that of
        `cores[-1]`.
      name: name of the module.

    Raises:
      ValueError: if `cores` is not an iterable, or if `skip_connections` is
          True and not all the modules are recurrent.
    """
    super(DeepRNN, self).__init__(name=name)

    if not isinstance(cores, collections.Iterable):
      raise ValueError("Cores should be an iterable object.")
    self._cores = tuple(cores)
    self._skip_connections = skip_connections
    self._concat_final_output_if_skip = concat_final_output_if_skip

    self._is_recurrent_list = [isinstance(core, rnn_core.RNNCore)
                               for core in self._cores]

    if self._skip_connections:
      tf.logging.log_first_n(
          tf.logging.WARN,
          "The `skip_connections` argument will be deprecated.",
          1
      )
      if not all(self._is_recurrent_list):
        raise ValueError("skip_connections are enabled but not all cores are "
                         "`snt.RNNCore`s, which is not supported. The following"
                         " cores were specified: {}.".format(self._cores))
      self._check_cores_output_sizes()

    self._num_recurrent = sum(self._is_recurrent_list)
    self._last_output_size = None

  def _check_cores_output_sizes(self):
    """Checks the output_sizes of the cores of the DeepRNN module.

    Raises:
      ValueError: if the outputs of the cores cannot be concatenated along their
        first dimension.
    """
    for core_sizes in zip(*tuple(_get_flat_core_sizes(self._cores))):
      first_core_list = core_sizes[0][1:]
      for i, core_list in enumerate(core_sizes[1:]):
        if core_list[1:] != first_core_list:
          raise ValueError("The outputs of the provided cores are not able "
                           "to be concatenated along the first feature "
                           "dimension. Core 0 has shape %s, whereas Core %d "
                           "has shape %s - these must only differ in the first "
                           "dimension" % (core_sizes[0], i + 1, core_list))

  def _build(self, inputs, prev_state):
    """Connects the DeepRNN module into the graph.

    If this is not the first time the module has been connected to the graph,
    the Tensors provided as input_ and state must have the same final
    dimension, in order for the existing variables to be the correct size for
    their corresponding multiplications. The batch size may differ for each
    connection.

    Args:
      inputs: a nested tuple of Tensors of arbitrary dimensionality, with at
        least an initial batch dimension.
      prev_state: a tuple of `prev_state`s that corresponds to the state
        of each one of the cores of the `DeepCore`.

    Returns:
      output: a nested tuple of Tensors of arbitrary dimensionality, with at
        least an initial batch dimension.
      next_state: a tuple of `next_state`s that corresponds to the updated state
        of each one of the cores of the `DeepCore`.

    Raises:
      ValueError: if connecting the module into the graph any time after the
        first time, and the inferred size of the inputs does not match previous
        invocations. This may happen if one connects a module any time after the
        first time that does not have the configuration of skip connections as
        the first time.
    """
    current_input = inputs
    next_states = []
    outputs = []
    recurrent_idx = 0
    concatenate = lambda *args: tf.concat(args, axis=-1)
    for i, core in enumerate(self._cores):
      if self._skip_connections and i > 0:
        current_input = nest.map_structure(concatenate, inputs, current_input)

      # Determine if this core in the stack is recurrent or not and call
      # accordingly.
      if self._is_recurrent_list[i]:
        current_input, next_state = core(current_input,
                                         prev_state[recurrent_idx])
        next_states.append(next_state)
        recurrent_idx += 1
      else:
        current_input = core(current_input)

      if self._skip_connections:
        outputs.append(current_input)

    if self._skip_connections and self._concat_final_output_if_skip:
      output = nest.map_structure(concatenate, *outputs)
    else:
      output = current_input

    self._last_output_size = _get_shape_without_batch_dimension(output)
    return output, tuple(next_states)

  def initial_state(self, batch_size, dtype=tf.float32, trainable=False,
                    trainable_initializers=None, trainable_regularizers=None,
                    name=None):
    """Builds the default start state for a DeepRNN.

    Args:
      batch_size: An int, float or scalar Tensor representing the batch size.
      dtype: The data type to use for the state.
      trainable: Boolean that indicates whether to learn the initial state.
      trainable_initializers: An initializer function or nested structure of
          functions with same structure as the `state_size` property of the
          core, to be used as initializers of the initial state variable.
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
      A tensor or nested tuple of tensors with same structure and shape as the
      `state_size` property of the core.

    Raises:
      ValueError: if the number of passed initializers is not the same as the
          number of recurrent cores.
    """
    initial_state = []
    if trainable_initializers is None:
      trainable_initializers = [None] * self._num_recurrent
    if trainable_regularizers is None:
      trainable_regularizers = [None] * self._num_recurrent

    num_initializers = len(trainable_initializers)

    if num_initializers != self._num_recurrent:
      raise ValueError("The number of initializers and recurrent cores should "
                       "be the same. Received %d initializers for %d specified "
                       "recurrent cores."
                       % (num_initializers, self._num_recurrent))

    with tf.name_scope(self._initial_state_scope(name)):
      recurrent_idx = 0
      for is_recurrent, core in zip(self._is_recurrent_list, self._cores):
        if is_recurrent:
          core_initial_state = core.initial_state(
              batch_size, dtype=dtype, trainable=trainable,
              trainable_initializers=trainable_initializers[recurrent_idx],
              trainable_regularizers=trainable_regularizers[recurrent_idx])
          initial_state.append(core_initial_state)
          recurrent_idx += 1
    return tuple(initial_state)

  @property
  def state_size(self):
    sizes = []
    for is_recurrent, core in zip(self._is_recurrent_list, self._cores):
      if is_recurrent:
        sizes.append(core.state_size)
    return tuple(sizes)

  @property
  def output_size(self):
    if self._skip_connections and self._concat_final_output_if_skip:
      output_size = []
      for core_sizes in zip(*tuple(_get_flat_core_sizes(self._cores))):
        added_core_size = core_sizes[0]
        added_core_size[-1] = sum([size[-1] for size in core_sizes])
        output_size.append(tf.TensorShape(added_core_size))
      return nest.pack_sequence_as(structure=self._cores[0].output_size,
                                   flat_sequence=output_size)
    else:
      # Assumes that an element of cores which does not have the output_size
      # property does not affect the output shape. Then the 'last' core in the
      # sequence with output_size information should be the output_size of the
      # DeepRNN. This heuristic is error prone, but we would lose a lot of
      # flexibility if we tried to enforce that the final core must have an
      # output_size field (e.g. it would be impossible to add a TF nonlinearity
      # as the final "core"), but we should at least print a warning if this
      # is the case.
      final_core = self._cores[-1]
      if hasattr(final_core, "output_size"):
        # This is definitely the correct value, so no warning needed.
        return final_core.output_size

      # If we have connected the module at least once, we can get the output
      # size of whatever was actually produced.
      if self._last_output_size is not None:
        tf.logging.warning(
            "Final core does not contain .output_size, but the "
            "DeepRNN has been connected into the graph, so inferred output "
            "size as %s", self._last_output_size)
        return self._last_output_size

      # If all else fails, iterate backwards through cores and return the
      # first one which has an output_size field. This can be incorrect in
      # various ways, so warn loudly.
      try:
        guessed_output_size = next(core.output_size
                                   for core in reversed(self._cores)
                                   if hasattr(core, "output_size"))
      except StopIteration:
        raise ValueError("None of the 'cores' have output_size information.")

      tf.logging.warning(
          "Trying to infer output_size of DeepRNN, but the final core %s does "
          "not have the .output_size field. The guessed output_size is %s "
          "but this may not be correct. If you see shape errors following this "
          "warning, you must change the cores used in the DeepRNN so that "
          "the final core used has a correct .output_size property.",
          final_core, guessed_output_size)
      return guessed_output_size


class ModelRNN(rnn_core.RNNCore):
  """RNNCore that ignores input and uses a model to compute its next state."""

  def __init__(self, model, name="model_rnn"):
    """Construct a Basic RNN core.

    Args:
      model: callable that computes the next state.
      name: name of the module.

    Raises:
      TypeError: if model is not a callable object or if it is an RNNCore.
      AttributeError: if model does not have an output_size attribute.
    """
    super(ModelRNN, self).__init__(name=name)

    if not callable(model):
      raise TypeError("Model must be callable.")
    if isinstance(model, rnn_core.RNNCore):
      raise TypeError("Model should not be an RNNCore.")

    try:
      self._output_size = model.output_size
    except AttributeError:
      raise AttributeError("Model should have an output_size attribute.")

    self._model = model

  def _build(self, inputs, prev_state):
    """Connects the ModelRNN module into the graph.

    If this is not the first time the module has been connected to the graph,
    the Tensors provided as input_ and state must have the same final
    dimension, in order for the existing variables to be the correct size for
    their corresponding multiplications. The batch size may differ for each
    connection.

    Args:
      inputs: Tensor input to the ModelRNN (ignored).
      prev_state: Tensor of size `model.output_size`.

    Returns:
      output: Tensor of size `model.output_size`.
      next_state: Tensor of size `model.output_size`.
    """
    next_state = self._model(prev_state)

    # For ModelRNN, the next state of the RNN is the same as the output
    return next_state, next_state

  @property
  def state_size(self):
    return self._output_size

  @property
  def output_size(self):
    return self._output_size


class BidirectionalRNN(base.AbstractModule):
  """Bidirectional RNNCore that processes the sequence forwards and backwards.

    Based upon the encoder implementation in: https://arxiv.org/abs/1409.0473

  This interface of this module is different than the typical ones found in
  the RNNCore family.  The primary difference is that it is pre-conditioned on
  the full input sequence in order to produce a full sequence of outputs and
  states concatenated along the feature dimension among the forward and
  backward cores.
  """

  def __init__(self, forward_core, backward_core, name="bidir_rnn"):
    """Construct a Bidirectional RNN core.

    Args:
      forward_core: callable RNNCore module that computes forward states.
      backward_core: callable RNNCore module that computes backward states.
      name: name of the module.

    Raises:
      ValueError: if not all the modules are recurrent.
    """
    super(BidirectionalRNN, self).__init__(name=name)
    self._forward_core = forward_core
    self._backward_core = backward_core
    def _is_recurrent(core):
      has_rnn_core_interface = (hasattr(core, "initial_state") and
                                hasattr(core, "output_size") and
                                hasattr(core, "state_size"))
      return isinstance(core, rnn_core.RNNCore) or has_rnn_core_interface
    if not(_is_recurrent(forward_core) and _is_recurrent(backward_core)):
      raise ValueError("Forward and backward cores must both be instances of"
                       "RNNCore.")

  def _build(self, input_sequence, state):
    """Connects the BidirectionalRNN module into the graph.

    Args:
      input_sequence: tensor (time, batch, [feature_1, ..]). It must be
          time_major.
      state: tuple of states for the forward and backward cores.

    Returns:
      A dict with forward/backard states and output sequences:

        "outputs":{
            "forward": ...,
            "backward": ...},
        "state": {
            "forward": ...,
            "backward": ...}

    Raises:
      ValueError: in case time dimension is not statically known.
    """
    input_shape = input_sequence.get_shape()
    if input_shape[0] is None:
      raise ValueError("Time dimension of input (dim 0) must be statically"
                       "known.")
    seq_length = int(input_shape[0])
    forward_state, backward_state = state

    # Lists for the forward backward output and state.
    output_sequence_f = []
    output_sequence_b = []

    # Forward pass over the sequence.
    with tf.name_scope("forward_rnn"):
      state = forward_state
      output_sequence_f = [
          self._forward_core(input_sequence[i, :,], state)
          for i in six.moves.range(seq_length)]
      output_sequence_f = nest.map_structure(
          lambda *vals: tf.stack(vals), *output_sequence_f)

    # Backward pass over the sequence.
    with tf.name_scope("backward_rnn"):
      state = backward_state
      output_sequence_b = [
          self._backward_core(input_sequence[i, :,], state)
          for i in six.moves.range(seq_length)]
      output_sequence_b = nest.map_structure(
          lambda *vals: tf.stack(vals), *output_sequence_b)

    # Compose the full output and state sequeneces.
    return {
        "outputs": {
            "forward": output_sequence_f[0],
            "backward": output_sequence_b[0]
        },
        "state": {
            "forward": output_sequence_f[1],
            "backward": output_sequence_b[1]
        }
    }

  def initial_state(self, batch_size, dtype=tf.float32, trainable=False,
                    trainable_initializers=None, trainable_regularizers=None,
                    name=None):
    """Builds the default start state for a BidirectionalRNN.

    The Bidirectional RNN flattens the states of its forward and backward cores
    and concatentates them.

    Args:
      batch_size: An int, float or scalar Tensor representing the batch size.
      dtype: The data type to use for the state.
      trainable: Boolean that indicates whether to learn the initial state.
      trainable_initializers: An initializer function or nested structure of
          functions with same structure as the `state_size` property of the
          core, to be used as initializers of the initial state variable.
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
      Tuple of initial states from forward and backward RNNs.
    """
    name = "state" if name is None else name
    forward_initial_state = self._forward_core.initial_state(
        batch_size, dtype, trainable, trainable_initializers,
        trainable_regularizers, name=name+"_forward")
    backward_initial_state = self._backward_core.initial_state(
        batch_size, dtype, trainable, trainable_initializers,
        trainable_regularizers, name=name+"_backward")
    return forward_initial_state, backward_initial_state

  @property
  def state_size(self):
    """Flattened state size of cores."""
    return self._forward_core.state_size, self._backward_core.state_size

  @property
  def output_size(self):
    """Flattened output size of cores."""
    return self._forward_core.output_size, self._backward_core.output_size
