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

"""Tensorflow op that scales gradient for backwards pass."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import function

tfe = tf.contrib.eager


def _scale_gradient_op(dtype):
  """Create an op that scales gradients using a Defun.

  The tensorflow Defun decorator creates an op and tensorflow caches these ops
  automatically according to `func_name`. Using a Defun decorator twice with the
  same `func_name` does not create a new op, instead the cached op is used.

  This method produces a new op the first time it is called with a given `dtype`
  argument, and then uses the cached op each time it is called after that with
  the same `dtype`. The scale value is given as an argument for the forward pass
  method so that it can be used in the backwards pass.

  Args:
    dtype: the dtype of the net whose gradient is being scaled.

  Returns:
    The op that scales gradients.
  """

  def scale_gradient_backward(op, grad):
    scale = op.inputs[1]
    scaled_grad = grad * scale
    return scaled_grad, None

  # Note that if the forward pass implementation involved the creation of ops,
  # _scale_gradient_op would require some memoization mechanism.
  def scale_gradient_forward(x, scale):
    del scale  # Unused.
    return x

  func_name = "ScaleGradient_{}".format(dtype.name)
  return function.Defun(
      dtype, dtype,
      python_grad_func=scale_gradient_backward,
      func_name=func_name)(scale_gradient_forward)


def scale_gradient(net, scale, name="scale_gradient"):
  """Scales gradients for the backwards pass.

  This might be used to, for example, allow one part of a model to learn at a
  lower rate than the rest.

  WARNING: Think carefully about how your optimizer works. If, for example, you
  use rmsprop, the gradient is always rescaled (with some additional epsilon)
  towards unity. This means `scale_gradient` won't have the effect of
  lowering the learning rate.

  If `scale` is `0.0`, this op reduces to `tf.stop_gradient`. If `scale`
  is `1.0`, this op reduces to `tf.identity`.

  Args:
    net: A `tf.Tensor` or in eager mode a callable that produces a `tf.Tensor`.
    scale: The scale factor for the gradient on the backwards pass.
    name: A name for the operation (optional).

  Returns:
    In graph mode returns a `tf.Tensor` with the same type as the input tensor.
    In eager mode returns a callable wrapping `net` whose gradients are scaled.

  Raises:
    ValueError: If `net` dtype is non-float and `scale` is not zero or one.
  """
  if tf.executing_eagerly():
    if not callable(net):
      raise ValueError(
          "In eager mode `net` must be a callable (similar to how optimizers "
          "must be used when executing eagerly).")
    return tfe.defun(lambda *a, **k: scale_gradient(net(*a, **k), scale, name))

  if scale == 0.0:
    return tf.stop_gradient(net, name=name)
  elif scale == 1.0:
    return tf.identity(net, name=name)
  else:
    if not net.dtype.is_floating:
      raise ValueError("clip_gradient does not support non-float `net` inputs.")

    with tf.name_scope(name, "scale_gradient", values=[net]):
      dtype = net.dtype.base_dtype  # Convert ref dtypes to regular dtypes.
      scale_tensor = tf.convert_to_tensor(scale, dtype=dtype)

      scale_gradient_op = _scale_gradient_op(dtype)
      output = scale_gradient_op(net, scale_tensor)
      output.set_shape(net.get_shape())

    return output
