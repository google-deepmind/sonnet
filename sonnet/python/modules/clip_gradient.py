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
  """

  def _clip_gradient_backward(unused_op, grad):
    return tf.clip_by_value(grad, clip_value_min, clip_value_max)

  @function.Defun(
      net.dtype,
      python_grad_func=_clip_gradient_backward,
      func_name="ClipGradient")
  def _clip_gradient_forward(x):
    return x

  with tf.name_scope(name, "clip_gradient", values=[net]):
    output = _clip_gradient_forward(net)
    output.set_shape(net.get_shape())

  return output
