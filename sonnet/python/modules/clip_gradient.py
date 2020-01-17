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

"""Tensorflow op that clips gradient for backwards pass."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf


@tf.custom_gradient
def _clip_gradient(x, clip_value_min, clip_value_max):
  def grad(dy):
    return tf.clip_by_value(dy, clip_value_min, clip_value_max), None, None
  return x, grad


def clip_gradient(net, clip_value_min, clip_value_max, name=None):
  """Clips respective gradients of a given tensor.

  Acts as identity for the forward pass, but clips gradient tensor element-wise
  by value during the backward pass. Any gradient values less than
  `clip_value_min` or greater than `clip_values_max` are set to the respective
  limit values.

  Args:
    net: A `tf.Tensor`.
    clip_value_min: A 0-D Tensor or scalar. The minimum value to clip by.
    clip_value_max: A 0-D Tensor or scalar. The maximum value to clip by.
    name: A name for the operation (optional, default 'clip_gradient').

  Returns:
    A `tf.Tensor` with the same type as the input tensor.

  Raises:
    ValueError: If `net` dtype is non-float.
  """
  if not net.dtype.is_floating:
    raise ValueError("clip_gradient does not support non-float `net` inputs.")

  with tf.name_scope(name, "clip_gradient", values=[net]):
    dtype = net.dtype.base_dtype  # Convert ref dtypes to regular dtypes.
    min_tensor = tf.convert_to_tensor(clip_value_min, dtype=dtype)
    max_tensor = tf.convert_to_tensor(clip_value_max, dtype=dtype)

    output = _clip_gradient(net, min_tensor, max_tensor)

  return output
