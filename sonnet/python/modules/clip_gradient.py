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

import tensorflow as tf
from tensorflow.python.framework import function


def _clip_gradient_op(dtype):
  """Create an op that clips gradients using a Defun.

  The tensorflow Defun decorator creates an op and tensorflow caches these op
  automatically according to `func_name`. Using a Defun decorator twice with the
  same `func_name` does not create a new op, instead the cached op is used.

  This method produces a new op the first time it is called with a given `dtype`
  argument, and then uses the cached op each time it is called after that with
  the same `dtype`. The min and max clip values are given as arguments for the
  forward pass method so that they can be used in the backwards pass.

  Args:
    dtype: the dtype of the net whose gradient is being clipped.

  Returns:
    The op that clips gradients.
  """

  def clip_gradient_backward(op, grad):
    clip_value_min = op.inputs[1]
    clip_value_max = op.inputs[2]
    clipped_grad = tf.clip_by_value(grad, clip_value_min, clip_value_max)
    return clipped_grad, None, None

  def clip_gradient_forward(x, clip_value_min, clip_value_max):
    del clip_value_min  # Unused.
    del clip_value_max  # Unused.
    return x

  func_name = "ClipGradient_{}".format(dtype.name)
  return function.Defun(
      dtype, dtype, dtype,
      python_grad_func=clip_gradient_backward,
      func_name=func_name)(clip_gradient_forward)


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

    clip_gradient_op = _clip_gradient_op(dtype)
    output = clip_gradient_op(net, min_tensor, max_tensor)
    output.set_shape(net.get_shape())

  return output
