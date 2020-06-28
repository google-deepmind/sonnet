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

import tensorflow.compat.v1 as tf1
import tensorflow as tf
import tree


def _check_inputs_dtype(inputs, expected_dtype):
  if inputs.dtype is not expected_dtype:
    raise TypeError("inputs must have dtype {!r}, got {!r}".format(
        expected_dtype, inputs.dtype))
  return expected_dtype


def _safe_where(condition, x, y):  # pylint: disable=g-doc-args
  """`tf.where` which allows scalar inputs."""
  if x.shape.rank == 0:
    # This is to match the `tf.nn.*_rnn` behavior. In general, we might
    # want to branch on `tf.reduce_all(condition)`.
    return y
  # TODO(tomhennigan) Broadcasting with SelectV2 is currently broken.
  return tf1.where(condition, x, y)



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