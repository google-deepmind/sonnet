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
    net: A `tf.Tensor`.
    scale: The scale factor for the gradient on the backwards pass.
    name: A name for the operation (optional).

  Returns:
    A `tf.Tensor` with the same type as the input tensor.
  """
  if scale == 0.0:
    return tf.stop_gradient(net, name=name)
  elif scale == 1.0:
    return tf.identity(net, name=name)
  else:
    scale_tensor = tf.convert_to_tensor(scale)

    @function.Defun(tf.float32, tf.float32,
                    python_grad_func=lambda op, g: (g * op.inputs[1], None),
                    func_name="ScaleGradient")
    def gradient_scaler(x, unused_scale):
      return x

    output = gradient_scaler(net, scale_tensor, name=name)  # pylint:disable=unexpected-keyword-arg
    output.set_shape(net.get_shape())

    return output
