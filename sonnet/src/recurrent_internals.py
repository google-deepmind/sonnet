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
"""Utils for Recurrent Neural Network cores."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import uuid

import tensorflow.compat.v1 as tf1
import tensorflow as tf
import tree

# pylint: disable=g-direct-tensorflow-import
# Required for specializing `UnrolledLSTM` per device.
from tensorflow.python import context as context_lib
from tensorflow.python.eager import function as function_lib
# pylint: enable=g-direct-tensorflow-import


LSTMState = collections.namedtuple("LSTMState", ["hidden", "cell"])


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


def _safe_where(condition, x, y):  # pylint: disable=g-doc-args
  """`tf.where` which allows scalar inputs."""
  if x.shape.rank == 0:
    # This is to match the `tf.nn.*_rnn` behavior. In general, we might
    # want to branch on `tf.reduce_all(condition)`.
    return y
  # TODO(tomhennigan) Broadcasting with SelectV2 is currently broken.
  return tf1.where(condition, x, y)


def _check_inputs_dtype(inputs, expected_dtype):
  if inputs.dtype is not expected_dtype:
    raise TypeError("inputs must have dtype {!r}, got {!r}".format(
        expected_dtype, inputs.dtype))
  return expected_dtype


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
      function_lib.register(functions[device], *args, **kwargs)
    return functions[default](*args, **kwargs)

  return wrapper
