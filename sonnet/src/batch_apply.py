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

from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

from sonnet.src import base
import tensorflow as tf
from typing import Optional, Text


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
               module: base.Module,
               num_dims: int = 2,
               name: Optional[Text] = None):
    super(BatchApply, self).__init__(name=name)
    self.module = module
    self.num_dims = num_dims

  def __call__(self, *args, **kwargs):
    example = first_leaf(args, kwargs)
    num_dims = self.num_dims
    merge = lambda x: merge_leading_dims(x, num_dims=num_dims)
    split = lambda x: split_leading_dims(x, num_dims=num_dims, inputs=example)

    # Merge leading dimensions of inputs.
    # Example: [T, B, N] -> [T*B, N]
    args = tf.nest.map_structure(merge, args)
    kwargs = tf.nest.map_structure(merge, kwargs)

    # Compute merged output.
    # Example: [T*B, O]
    outputs = self.module(*args, **kwargs)

    # Split leading dimensions of output to match input.
    # Example: [T*B, O] -> [T, B, O]
    return tf.nest.map_structure(split, outputs)


def first_leaf(args, kwargs):
  flat_args = tf.nest.flatten(args)
  if flat_args:
    return flat_args[0]
  flat_kwargs = tf.nest.flatten(kwargs)
  if flat_kwargs:
    return flat_kwargs[0]
  return None


def split_leading_dims(
    x: Optional[tf.Tensor],
    inputs: tf.Tensor,
    num_dims: int,
) -> Optional[tf.Tensor]:
  if x is None:
    return x
  out_shape = inputs.shape[:num_dims] + x.shape[1:]
  return tf.reshape(x, out_shape)


def merge_leading_dims(
    x: Optional[tf.Tensor],
    num_dims: int,
) -> Optional[tf.Tensor]:
  """Merges leading dimensions."""
  if x is None or not isinstance(x, (tf.Tensor, tf.Variable)):
    return x

  if len(x.shape) < num_dims:
    return x
  return tf.reshape(x, [-1] + x.shape.as_list()[num_dims:])
