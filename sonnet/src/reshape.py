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

from typing import Optional, Sequence

import numpy as np
from sonnet.src import base
from sonnet.src import once
from sonnet.src import types
import tensorflow as tf


def reshape(inputs: tf.Tensor,
            output_shape: types.ShapeLike,
            preserve_dims: int = 1,
            name: Optional[str] = None) -> tf.Tensor:
  """A shortcut for applying :class:`Reshape` to the ``inputs``."""
  return Reshape(output_shape, preserve_dims, name=name)(inputs)


def flatten(inputs: tf.Tensor, name: str = "flatten") -> tf.Tensor:
  """A shortcut for applying :class:`Flatten` to the ``inputs``."""
  return Flatten(name=name)(inputs)


def _infer_shape(output_shape: types.ShapeLike, dimensions: Sequence[int]):
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

  For example, given an input tensor with shape ``[B, H, W, C, D]``::

      >>> B, H, W, C, D = range(1, 6)
      >>> x = tf.ones([B, H, W, C, D])

  The default behavior when ``output_shape`` is ``(-1, D)`` is to flatten
  all dimensions between ``B`` and ``D``::

      >>> mod = snt.Reshape(output_shape=(-1, D))
      >>> assert mod(x).shape == [B, H*W*C, D]

  You can change the number of preserved leading dimensions via
  ``preserve_dims``::

      >>> mod = snt.Reshape(output_shape=(-1, D), preserve_dims=2)
      >>> assert mod(x).shape == [B, H, W*C, D]

      >>> mod = snt.Reshape(output_shape=(-1, D), preserve_dims=3)
      >>> assert mod(x).shape == [B, H, W, C, D]

      >>> mod = snt.Reshape(output_shape=(-1, D), preserve_dims=4)
      >>> assert mod(x).shape == [B, H, W, C, 1, D]
  """

  def __init__(self,
               output_shape: types.ShapeLike,
               preserve_dims: int = 1,
               name: Optional[str] = None):
    """Constructs a ``Reshape`` module.

    Args:
      output_shape: Shape to reshape the input tensor to while preserving its
        first ``preserve_dims` dimensions. When the special value -1 appears in
        ``output_shape`` the corresponding size is automatically inferred. Note
        that -1 can only appear once in ``output_shape``.
        To flatten all non-batch dimensions use :class:`Flatten`.
      preserve_dims: Number of leading dimensions that will not be reshaped.
      name: Name of the module.

    Raises:
      ValueError: If ``preserve_dims`` is not positive.
    """
    super().__init__(name=name)

    if preserve_dims <= 0:
      raise ValueError("Argument preserve_dims should be >= 1.")

    self._output_shape = output_shape
    self._preserve_dims = preserve_dims

  @once.once
  def _initialize(self, inputs: tf.Tensor):
    if inputs.shape.rank < self._preserve_dims:
      raise ValueError("Input tensor has {} dimensions, should have at least "
                       "as many as preserve_dims={}".format(
                           inputs.shape.rank, self._preserve_dims))

    self._input_shape = inputs.shape

  def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
    """Reshapes ``inputs``.

    Args:
      inputs: A tensor of shape ``[b_1, b_2, ..., b_preserve_dims,
        b_preserve_dims + 1, ...]``.

    Returns:
      A tensor of shape
        ``[b_1, b_2, ..., b_preserve_dims, b_reshape_1, b_reshape_2, ...]``,
        with reshaping defined by the constructor ``output_shape`` parameter.

    Raises:
      ValueError: If ``output_shape`` is incompatible with shape of the
        ``inputs``; or if ``output_shape`` contains more than one wildcard -1;
        or if the ``inputs`` rank is less than ``preserved_dims``; or if
        the ``inputs`` shape contains unknown, non-preserved dimensions
        (except when the unknown dimension is the only non-preserved
        dimension and doesn't actually need reshaping).
    """
    self._initialize(inputs)

    # Resolve the wildcard if any.
    output_shape = tuple(self._output_shape)
    if -1 in output_shape:
      reshaped_shape = inputs.shape[self._preserve_dims:]
      if reshaped_shape.is_fully_defined():
        output_shape = _infer_shape(output_shape, reshaped_shape)

    preserved_shape = inputs.shape[:self._preserve_dims]
    if preserved_shape.is_fully_defined():
      output = tf.reshape(inputs, tuple(preserved_shape) + output_shape)
    else:
      dynamic_preserved_shape = tf.shape(inputs)[:self._preserve_dims]
      output = tf.reshape(
          inputs, tf.concat([dynamic_preserved_shape, output_shape], axis=0))
    return output

  @base.no_name_scope
  def reversed(self, name: Optional[str] = None) -> "Reshape":
    """Returns inverse batch reshape."""
    if name is None:
      name = self.name + "_reversed"

    return Reshape(
        output_shape=self._input_shape[self._preserve_dims:],
        preserve_dims=self._preserve_dims,
        name=name)


class Flatten(Reshape):
  """Flattens the input Tensor, preserving the batch dimension(s).

  ``Flatten`` reshapes input tensors to combine all trailing dimensions
  apart from the first. Additional leading dimensions can be preserved
  by setting the ``preserve_dims`` parameter.

  See :class:`Reshape` for more details.
  """

  def __init__(self, preserve_dims: int = 1, name: Optional[str] = None):
    """Constructs a ``Flatten`` module.

    Args:
      preserve_dims: Number of leading dimensions that will not be reshaped.
      name: Name of the module.
    """
    super().__init__(output_shape=(-1,), preserve_dims=preserve_dims, name=name)
