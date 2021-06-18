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
"""Merges a number of leading dimensions of an input tensor to manipulate it."""

from typing import Any, Callable, Optional, Sequence, Union

import numpy as np
from sonnet.src import base
import tensorflow as tf
import tree


class BatchApply(base.Module):
  """Merges a number of leading dimensions of an input tensor to manipulate it.

  Merges a number of leading dimensions of a tensor into a single dimension,
  connects the provided module, then splits the leading dimension of the
  result to match the input.

  Input tensors whose rank is smaller than the number of dimensions to collapse
  (e.g. all scalar values, which are tensors of rank 0), are passed unaltered to
  the provided module.

  This is useful for applying some module to each timestep of a Time x Batch x N
  tensor. If a module is hard coded to only support 2D (Batch x N) then the
  full 3D Tensor cannot be provided. BatchApply will 'merge' the first two
  dimensions of the sequence tensor by reshaping to a (Time * Batch) x N Tensor,
  and then the internal module can be applied. The result of that operation is
  reshaped such that its first dimensions are split to match the leading
  dimensions of the input.
  """

  def __init__(self,
               module: Callable[..., tf.Tensor],
               num_dims: int = 2,
               name: Optional[str] = None):
    super().__init__(name=name)
    self.module = module
    self.num_dims = num_dims

  def __call__(self, *args, **kwargs):
    example = first_leaf(args, kwargs)
    if example is None:
      raise ValueError("BatchApply requires at least one tensor input.")

    num_dims = self.num_dims
    merge = lambda x: merge_leading_dims(x, num_dims=num_dims)
    split = lambda x: split_leading_dim(x, num_dims=num_dims, example=example)

    # Merge leading dimensions of inputs.
    # Example: [T, B, N] -> [T*B, N]
    args = tree.map_structure(merge, args)
    kwargs = tree.map_structure(merge, kwargs)

    # Compute merged output.
    # Example: [T*B, O]
    outputs = self.module(*args, **kwargs)

    # Split leading dimensions of output to match input.
    # Example: [T*B, O] -> [T, B, O]
    return tree.map_structure(split, outputs)


def first_leaf(args, kwargs) -> Optional[Any]:
  flat_args = tree.flatten(args)
  if flat_args:
    return flat_args[0]
  flat_kwargs = tree.flatten(kwargs)
  if flat_kwargs:
    return flat_kwargs[0]
  return None


def split_leading_dim(
    x: Optional[tf.Tensor],
    example: tf.Tensor,
    num_dims: int,
) -> Optional[tf.Tensor]:
  """Split the first dimension of a tensor to match an example.

  See :func:`merge_leading_dims`.

  >>> x = tf.ones([6, 1])
  >>> example = tf.ones([3, 2, 1])
  >>> snt.split_leading_dim(x, example, 2)
  <tf.Tensor: ...shape=(3, 2, 1), ...>

  If ``x`` is not a :tf:`Tensor` or :tf:`Variable` then is is returned
  unchanged:

  >>> snt.split_leading_dim('not a tensor', example, 2)
  'not a tensor'

  Args:
    x: A tensor with leading dim merged.
    example: An Tensor with leading dim not merged.
    num_dims: The number of leading dimensions of example to use.

  Returns:
    A tensor with leading dim split, or the input unchanged.
  """
  if x is None or not isinstance(x, (tf.Tensor, tf.Variable)):
    return x

  static_shape = example.shape[:num_dims] + x.shape[1:]
  if static_shape.is_fully_defined():  # pytype: disable=attribute-error
    return tf.reshape(x, static_shape)

  # Shape can't be inferred statically.
  leading_dims = tf.shape(example)[:num_dims]
  other_dims = tf.shape(x)[1:]
  dynamic_shape = tf.concat([leading_dims, other_dims], axis=0)
  return tf.reshape(x, dynamic_shape)


def maybe_prod(s: Sequence[Union[int, None]]) -> Optional[int]:
  try:
    return np.prod(s)
  except TypeError:
    # Can happen if the input contains `None`.
    return None


def merge_leading_dims(
    x: Optional[tf.Tensor],
    num_dims: int,
) -> Optional[tf.Tensor]:
  """Merges leading dimensions of a tensor.

  See :func:`split_leading_dim`.

  >>> x = tf.ones([3, 2, 1])
  >>> snt.merge_leading_dims(x, num_dims=2)
  <tf.Tensor: ...shape=(6, 1), ...>

  If the rank of ``x`` is less than ``num_dims`` it is returned unchanged:

  >>> snt.merge_leading_dims(x, 4)
  <tf.Tensor: ...shape=(3, 2, 1), ...>

  If ``x`` is not a :tf:`Tensor` or :tf:`Variable` then is is returned
  unchanged:

  >>> snt.merge_leading_dims('not a tensor', 1)
  'not a tensor'

  Args:
    x: A :tf:`Tensor` to merge.
    num_dims: The number of leading dimensions to merge.

  Returns:
    A :tf:`Tensor` with merged leading dimensions or the input unchanged.
  """
  if x is None or not isinstance(x, (tf.Tensor, tf.Variable)):
    return x

  # Check if the rank of the input tensor is well-defined.
  if x.shape.dims is None:
    raise ValueError(
        "Can't merge leading dimensions of tensor of unknown rank.")

  # We can only merge the num_dims leading dimensions if the rank of the given
  # tensor is sufficiently large.
  if num_dims > x.shape.rank:
    return x

  static_shape = [maybe_prod(x.shape[:num_dims])] + x.shape[num_dims:]
  if static_shape.is_fully_defined():  # pytype: disable=attribute-error
    return tf.reshape(x, static_shape)

  # Shape can't be inferred statically.
  tensor_shape = tf.shape(x)
  leading_dim = tf.reduce_prod(tensor_shape[:num_dims], keepdims=True)
  other_dims = tensor_shape[num_dims:]
  dynamic_shape = tf.concat([leading_dim, other_dims], axis=0)
  result = tf.reshape(x, dynamic_shape)
  # We lose some static shape information from the above reduce/slice/concat
  # dance, so we explicitly pass it in from what we computed earlier.
  result.set_shape(static_shape)
  return result
