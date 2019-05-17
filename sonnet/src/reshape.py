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

"""Reshaping Sonnet modules."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sonnet.src import base
import tensorflow as tf


def _extract_input_shape(inputs, preserve_dims=1):
  """Extracts the shape minus ignored dimensions from `inputs`.

      >>> _extract_input_shape(tf.ones([1, 2, 3]))
      [2, 3]
      >>> _extract_input_shape(tf.ones([1, 2, 3]), preserve_dims=2)
      [3]

  Args:
    inputs: A tf.Tensor whose shape should be extracted.
    preserve_dims: Number of leading dimensions that will not be discarded.

  Returns:
    A list with the preserved dimensions from the shape of `inputs`.

  Raises:
    ValueError: If the number of dimensions in the input is not compatible with
        preserve dims.
  """
  input_shape = inputs.get_shape().as_list()
  if len(input_shape) < preserve_dims:
    raise ValueError("Input tensor has {} dimensions, should have at least "
                     "as many as preserve_dims={}".format(
                         len(input_shape), preserve_dims))
  return input_shape[preserve_dims:]


def _batch_reshape(inputs, output_shape, preserve_dims=1):
  """Reshapes input Tensor, preserving the batch dimension.

      >>> _batch_reshape(
      ...   tf.ones([10, 2, 2, 3]), output_shape=[-1], preserve_dims=1)
      <tf.Tensor: ... shape=(10, 12), ...>

      >>> _batch_reshape(
      ...   tf.ones([10, 2, 2, 3]), output_shape=[-1], preserve_dims=2)
      <tf.Tensor: ... shape=(10, 2, 6), ...>

  Args:
    inputs: A Tensor of shape
        `[b_1, b_2, ..., b_preserve_dims, b_preserve_dims+1, ...]`.
    output_shape: Shape to reshape the input Tensor to while preserving its
        first `preserve_dims` dimensions; `shape` can be either a tuple/list, or
        a callable that returns the actual shape. The callable does not need to
        be ready to return something meaningful at construction time, but it
        will be required to be able to do so when the module is connected to the
        graph. When the special value -1 appears in `shape` the corresponding
        size is automatically inferred. Note that -1 can only appear once in
        `shape`. To flatten all non-batch dimensions, the `snt.Flatten` module
        can also be used.
    preserve_dims: Number of leading dimensions that will not be reshaped. For
        example, given an input Tensor with shape `[B, H, W, C, D]`, and
        argument `shape` equal to `(-1, D)`:
        * `preserve_dims=1` returns a Tensor with shape `[B, H*W*C, D]`.
        * `preserve_dims=2` returns a Tensor with shape `[B, H, W*C, D]`.
        * `preserve_dims=3` returns a Tensor with shape `[B, H, W, C, D]`.
        * `preserve_dims=4` returns a Tensor with shape `[B, H, W, C, 1, D]`.
        * `preserve_dims>=5` will throw an error on build unless D=1.
        The preserved dimensions can be unknown at building time.

  Returns:
    A Tensor of shape
        `[b_1, b_2, ..., b_preserve_dims, b_reshape_1, b_reshape_2, ...]`,
        with reshaping defined by the constructor `shape` parameter.

  Raises:
    ValueError: If output shape is incompatible with input shape; or if
        shape array contains non numeric entries; or if shape array contains
        more than 1 wildcard -1; or if the input array contains unknown,
        non-preserved dimensions (except when the unknown dimension is the
        only non-preserved dimension and doesn't actually need reshaping).
  """
  input_shape = _extract_input_shape(inputs, preserve_dims)

  # Special-case of 1 non-preserved dimension, where no reshape is necessary.
  # This is useful if the non-preserved dimension of `inputs` is unknown
  # at build time.
  if len(input_shape) == 1 and len(output_shape) == 1:
    if output_shape[0] == -1 or output_shape[0] == input_shape[0]:
      return inputs

  # Slicing the shape tensor loses information, we keep it in a list too.
  preserved_shape = tf.shape(inputs)[:preserve_dims]
  preserved_shape_list = inputs.get_shape()[:preserve_dims]

  if -1 in output_shape:
    trailing_shape = _infer_shape(output_shape, input_shape)
  else:
    trailing_shape = output_shape

  output_shape = tf.concat([preserved_shape, trailing_shape], 0)
  output = tf.reshape(inputs, output_shape)

  # Include shape information that was lost when we sliced the shape tensor.
  output.set_shape(preserved_shape_list.concatenate(trailing_shape))

  return output


def _infer_shape(output_shape, dimensions):
  """Replaces the -1 wildcard in the output shape vector.

  This function infers the correct output shape given the input dimensions.

  Args:
    output_shape: Output shape.
    dimensions: List of input non-batch dimensions.

  Returns:
    Tuple of non-batch output dimensions.
  """
  # Size of input.
  n = np.prod(dimensions)
  # Size of output where defined.
  v = np.array(output_shape)
  m = abs(np.prod(v))
  # Replace wildcard.
  v[v == -1] = n // m
  return tuple(v)


class Reshape(base.Module):
  """Reshapes input Tensor, preserving the batch dimension.

  For example, given an input Tensor with shape `[B, H, W, C, D]`:

      >>> B, H, W, C, D = range(1, 6)
      >>> x = tf.ones([B, H, W, C, D])

  The default behavior when `output_shape` is (-1, D) is to flatten all
  dimensions between `B` and `D`:

      >>> mod = snt.Reshape(output_shape=(-1, D))
      >>> assert mod(x).shape == [B, H*W*C, D]

  You can change the number of preserved leading dimensions via
  `preserve_dims`:

      >>> mod = snt.Reshape(output_shape=(-1, D), preserve_dims=2)
      >>> assert mod(x).shape == [B, H, W*C, D]

      >>> mod = snt.Reshape(output_shape=(-1, D), preserve_dims=3)
      >>> assert mod(x).shape == [B, H, W, C, D]

      >>> mod = snt.Reshape(output_shape=(-1, D), preserve_dims=4)
      >>> assert mod(x).shape == [B, H, W, C, 1, D]
  """

  def __init__(self, output_shape, preserve_dims=1, name=None):
    """Constructs a Reshape module.

    Args:
      output_shape: Shape to reshape the input Tensor to while preserving its
          first `preserve_dims` dimensions. When the special value -1
          appears in `output_shape` the corresponding size is automatically
          inferred. Note that -1 can only appear once in `output_shape`. To
          flatten all non-batch dimensions use `snt.Flatten`.
      preserve_dims: Number of leading dimensions that will not be reshaped.
      name: Name of the module.

    Raises:
      ValueError: If `preserve_dims <= 0`.
    """
    super(Reshape, self).__init__(name=name)

    if preserve_dims <= 0:
      raise ValueError("Argument preserve_dims should be >= 1.")

    self._output_shape = output_shape
    self._preserve_dims = preserve_dims

  def __call__(self, inputs):
    """Reshapes `inputs`.

    Args:
      inputs: A Tensor of shape
          `[b_1, b_2, ..., b_preserve_dims, b_preserve_dims + 1, ...]`.

    Returns:
      A Tensor of shape
          `[b_1, b_2, ..., b_preserve_dims, b_reshape_1, b_reshape_2, ...]`,
          with reshaping defined by the constructor `shape` parameter.

    Raises:
      ValueError: If output shape is incompatible with input shape; or if
          shape array contains non numeric entries; or if shape array contains
          more than 1 wildcard -1; or if the input array contains unknown,
          non-preserved dimensions (except when the unknown dimension is the
          only non-preserved dimension and doesn't actually need reshaping).
    """
    self.input_shape = _extract_input_shape(inputs, self._preserve_dims)

    return _batch_reshape(inputs,
                          output_shape=self._output_shape,
                          preserve_dims=self._preserve_dims)

  @base.no_name_scope
  def reversed(self, name=None):
    """Returns inverse batch reshape."""
    if name is None:
      name = self.name + "_reversed"

    return Reshape(output_shape=self.input_shape,
                   preserve_dims=self._preserve_dims,
                   name=name)


class Flatten(Reshape):
  """Flattens the input Tensor, preserving the batch dimension(s).

  `Flatten` reshapes input tensors to combine all trailing dimensions apart
  from the first. Additional leading dimensions can be preserved by setting the
  `preserve_dims` parameter.

  See `snt.Reshape` for more details.
  """

  def __init__(self, preserve_dims=1, name=None):
    """Constructs a Flatten module.

    Args:
      preserve_dims: Number of leading dimensions that will not be reshaped.
      name: Name of the module.
    """
    super(Flatten, self).__init__(
        output_shape=(-1,), preserve_dims=preserve_dims, name=name)
