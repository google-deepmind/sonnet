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

import tensorflow.compat.v1 as tf
from tensorflow.contrib.eager.python import tfe as contrib_eager

tfe = contrib_eager


@tf.custom_gradient
def _scale_gradient(x, scale):
  grad = lambda dy: (dy * scale, None)
  return x, grad


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
      output = _scale_gradient(net, scale_tensor)

    return output
