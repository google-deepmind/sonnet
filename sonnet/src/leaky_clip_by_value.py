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
"""Clipping operation with customized gradients."""

from typing import Optional

import tensorflow as tf


@tf.custom_gradient
def leaky_clip_by_value(t: tf.Tensor,
                        clip_value_min: tf.Tensor,
                        clip_value_max: tf.Tensor,
                        name: Optional[str] = None):
  """Clips tensor values to a specified min and max.

  The gradient is set to zero when tensor values are already out of bound and
  gradient-descent will push them even further away from the valid range. If
  gradient-descent pushes the values towards the valid range, the gradient will
  pass through without change.
  Note that this is assuming a gradient flow for minimization. For
  maximization, flip the gradient before it back-propagates to this op.

  Args:
    t: A Tensor.
    clip_value_min: A 0-D (scalar) Tensor, or a Tensor with the same shape as t.
      The minimum value to clip by.
    clip_value_max: A 0-D (scalar) Tensor, or a Tensor with the same shape as t.
      The maximum value to clip by.
    name: A name for the operation (optional).

  Returns:
    A clipped Tensor.

  Raises:
    ValueError: If the clip tensors would trigger array broadcasting that would
    make the returned tensor larger than the input.
  """
  clip_t = tf.clip_by_value(t, clip_value_min, clip_value_max, name=name)

  def grad(dy):
    """Custom gradient."""
    zeros = tf.zeros_like(dy)
    condition = tf.logical_or(
        tf.logical_and(t < clip_value_min, dy > 0),
        tf.logical_and(t > clip_value_max, dy < 0),
    )
    dy = tf.where(condition, zeros, dy)
    return dy, None, None

  return clip_t, grad
