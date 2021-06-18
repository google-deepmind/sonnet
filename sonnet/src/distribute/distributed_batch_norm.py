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
"""Distributed batch normalization module."""

from typing import Optional, Tuple

from sonnet.src import batch_norm
from sonnet.src import initializers
from sonnet.src import metrics
from sonnet.src import once
from sonnet.src import types

import tensorflow as tf


class CrossReplicaBatchNorm(batch_norm.BaseBatchNorm):
  """Cross-replica Batch Normalization.

  At every step the full batch is used to calculate the batch statistics even
  within a distributed setting (note only with ``snt.(Tpu)Replicator``).

  See :class:`BaseBatchNorm` for details.

  Attributes:
    scale: If ``create_scale=True``, a trainable :tf:`Variable` holding the
      current scale after the module is connected for the first time.
    offset: If ``create_offset``, a trainable :tf:`Variable` holding the current
      offset after the module is connected for the first time.
  """

  def __init__(self,
               create_scale: bool,
               create_offset: bool,
               moving_mean: metrics.Metric,
               moving_variance: metrics.Metric,
               eps: types.FloatLike = 1e-5,
               scale_init: Optional[initializers.Initializer] = None,
               offset_init: Optional[initializers.Initializer] = None,
               data_format: str = "channels_last",
               name: Optional[str] = None):
    """Constructs a ``CrossReplicaBatchNorm`` module.

    Args:
      create_scale: whether to create a trainable scale per channel applied
        after the normalization.
      create_offset: whether to create a trainable offset per channel applied
        after normalization and scaling.
      moving_mean: An object which keeps track of the moving average of the mean
        which can be used to normalize at test time. This object must have an
        update method which takes a value and updates the internal state and a
        value property which returns the current mean.
      moving_variance: An object which keeps track of the moving average of the
        variance which can be used to normalize at test time. This object must
        have an update method which takes a value and updates the internal state
        and a value property which returns the current variance.
      eps: Small epsilon to avoid division by zero variance. Defaults to
        ``1e-5``.
      scale_init: Optional initializer for the scale variable. Can only be set
        if ``create_scale=True``. By default scale is initialized to ``1``.
      offset_init: Optional initializer for the offset variable. Can only be set
        if ``create_offset=True``. By default offset is initialized to ``0``.
      data_format: The data format of the input. Can be either
        ``channels_first``, ``channels_last``, ``N...C`` or ``NC...``. By
        default it is ``channels_last``.
      name: Name of the module.
    """
    super().__init__(
        create_scale=create_scale,
        create_offset=create_offset,
        moving_mean=moving_mean,
        moving_variance=moving_variance,
        eps=eps,
        scale_init=scale_init,
        offset_init=offset_init,
        data_format=data_format,
        name=name)

  @once.once
  def _initialize(self, inputs: tf.Tensor):
    super()._initialize(inputs)

    # Always use the unfused op here as mean/var are calculated before the op is
    # called so no speed-up is gained from the fused op
    self._fused = False

  def _moments(self, inputs: tf.Tensor,
               use_batch_stats: types.BoolLike) -> Tuple[tf.Tensor, tf.Tensor]:
    replica_context = tf.distribute.get_replica_context()
    if replica_context is None:
      raise TypeError(
          "Cross replica batch norm cannot be called in cross-replica context.")

    if use_batch_stats:
      # Note: This uses var=E(x^2) - E(x)^2 instead of the more numerically
      # stable var=E((x-E(x))^2) as this means that with XLA the all_reduces can
      # be combined and a fusion removed giving significant speed-up.
      # If you see NaNs in your model please try the alternative formula and
      # file a bug with your use-case.
      mean = tf.reduce_mean(inputs, self._axis, keepdims=True)
      mean = replica_context.all_reduce("MEAN", mean)
      mean_of_squares = tf.reduce_mean(
          tf.square(inputs), self._axis, keepdims=True)
      mean_of_squares = replica_context.all_reduce("MEAN", mean_of_squares)
      mean_squared = tf.square(mean)
      var = mean_of_squares - mean_squared
      return mean, var

    else:  # use moving statistics
      mean = self.moving_mean.value
      variance = self.moving_variance.value
      return mean, variance
