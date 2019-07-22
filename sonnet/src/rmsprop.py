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

"""RMSProp module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
from sonnet.src import base
from sonnet.src import once
from sonnet.src import optimizer_utils
from sonnet.src import utils
import tensorflow as tf


class RMSProp(base.Optimizer):
  """RMSProp module.

  See: http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf

  Maintain a moving (discounted) average of the square of updates. Divides each
  update by the root of this average.

      ms <- decay * ms + (1-decay) * update^2
      mom <- momentum * mom + learning_rate * update / sqrt(ms + epsilon)
      parameter <- parameter - mom

  This implementation of `RMSprop` uses plain momentum, not Nesterov momentum.

  The centered version additionally maintains a moving average of the
  gradients, and uses that average to estimate the variance:

      mg <- decay * mg + (1-decay) * update
      ms <- decay * ms + (1-decay) * update^2
      mom <- momentum * mom + learning_rate * update / sqrt(ms - mg^2 + epsilon)
      parameter <- parameter - mom

  Attributes:
    learning_rate: Learning rate.
    decay: Learning rate decay over each update.
    momentum: Momentum scalar.
    epsilon: Small value to avoid zero denominator.
    centered: `True` if centered.
    mom: Accumulated mom for each parameter.
    ms: Accumulated ms for each parameter.
    mg: Accumulated mg for each parameter.
  """

  def __init__(self, learning_rate, decay=0.9, momentum=0.0, epsilon=1e-10,
               centered=False, name=None):
    """Constructs an `RMSProp` module.

    Args:
      learning_rate: Learning rate.
      decay: Learning rate decay over each update.
      momentum: Momentum scalar.
      epsilon: Small value to avoid zero denominator.
      centered: If True, gradients are normalized by the estimated variance of
        the gradient; if False, by the uncentered second moment. Setting this to
        True may help with training, but is slightly more expensive in terms of
        computation and memory. Defaults to False.
      name: Name for this module.
    """
    super(RMSProp, self).__init__(name)
    self.learning_rate = learning_rate
    self.decay = decay
    self.momentum = momentum
    self.epsilon = epsilon
    self.centered = centered
    self.mom = []
    self.ms = []
    self.mg = []

  @once.once
  def _initialize(self, parameters):
    zero_var = lambda p: utils.variable_like(p, trainable=False)
    with tf.name_scope("momentum"):
      self.mom.extend(zero_var(p) for p in parameters)
    with tf.name_scope("rms"):
      self.ms.extend(zero_var(p) for p in parameters)
    if self.centered:
      with tf.name_scope("mg"):
        self.mg.extend(zero_var(p) for p in parameters)

  def apply(self, updates, parameters):
    """Applies updates to parameters.

    Args:
      updates: A list of updates to apply to parameters. An update can be a
        `Tensor`, `IndexedSlice`, or `None`. Updates are often gradients, as
        returned by `tf.GradientTape.gradient`.
      parameters: A list of parameters. A parameter is a `tf.Variable`.

    Raises:
      ValueError: If `updates` and `parameters` are empty, have different
        lengths, or have inconsistent types.
    """
    optimizer_utils.check_distribution_strategy()
    optimizer_utils.check_updates_parameters(updates, parameters)
    self._initialize(parameters)
    for update, parameter, mom, ms, mg in six.moves.zip_longest(
        updates, parameters, self.mom, self.ms, self.mg):
      if update is not None:
        optimizer_utils.check_same_dtype(update, parameter)
        lr = tf.cast(self.learning_rate, update.dtype)
        decay = tf.cast(self.decay, update.dtype)
        mu = tf.cast(self.momentum, update.dtype)
        epsilon = tf.cast(self.epsilon, update.dtype)
        if isinstance(update, tf.IndexedSlices):
          update, indices = optimizer_utils.deduplicate_indexed_slices(
              update.values, update.indices)
          sparse_ms_update = (tf.square(update) * (1. - decay) +
                              ms.sparse_read(indices) * decay)
          ms.scatter_update(tf.IndexedSlices(sparse_ms_update, indices))
          if self.centered:
            sparse_mg_update = (update * (1. - decay) +
                                mg.sparse_read(indices) * decay)
            mg.scatter_update(tf.IndexedSlices(sparse_mg_update, indices))
            denominator = (
                sparse_ms_update - tf.square(sparse_mg_update) + epsilon)
          else:
            denominator = sparse_ms_update + epsilon
          sparse_mom_update = (mu * mom.sparse_read(indices)) + (
              lr * update * tf.math.rsqrt(denominator))
          mom.scatter_update(tf.IndexedSlices(sparse_mom_update, indices))
          parameter.scatter_sub(tf.IndexedSlices(sparse_mom_update, indices))
        else:
          ms.assign(tf.square(update) * (1. - decay) + ms * decay)
          if self.centered:
            mg.assign(update * (1. - decay) + mg * decay)
            denominator = ms - tf.square(mg) + epsilon
          else:
            denominator = ms + epsilon
          mom.assign((mu * mom) + (lr * update * tf.math.rsqrt(denominator)))
          parameter.assign_sub(mom)


class FastRMSProp(base.Optimizer):
  """RMSProp module."""

  def __init__(self, learning_rate, decay=0.9, momentum=0.0, epsilon=1e-10,
               centered=False, name=None):
    """Constructs an `RMSProp` module."""
    super(FastRMSProp, self).__init__(name)
    self.learning_rate = learning_rate
    self.decay = decay
    self.momentum = momentum
    self.epsilon = epsilon
    self.centered = centered
    self.mom = []
    self.ms = []
    self.mg = []

  @once.once
  def _initialize(self, parameters):
    zero_var = lambda p: utils.variable_like(p, trainable=False)
    with tf.name_scope("momentum"):
      self.mom.extend(zero_var(p) for p in parameters)
    with tf.name_scope("rms"):
      self.ms.extend(zero_var(p) for p in parameters)
    if self.centered:
      with tf.name_scope("mg"):
        self.mg.extend(zero_var(p) for p in parameters)

  def apply(self, updates, parameters):
    """Applies updates to parameters."""
    optimizer_utils.check_distribution_strategy()
    optimizer_utils.check_updates_parameters(updates, parameters)
    self._initialize(parameters)
    for update, parameter, mom, ms, mg in six.moves.zip_longest(
        updates, parameters, self.mom, self.ms, self.mg):
      if update is not None:
        optimizer_utils.check_same_dtype(update, parameter)
        learning_rate = tf.cast(self.learning_rate, update.dtype)
        decay = tf.cast(self.decay, update.dtype)
        momentum = tf.cast(self.momentum, update.dtype)
        epsilon = tf.cast(self.epsilon, update.dtype)
        if isinstance(update, tf.IndexedSlices):
          update, indices = optimizer_utils.deduplicate_indexed_slices(
              update.values, update.indices)
          if self.centered:
            tf.raw_ops.ResourceSparseApplyCenteredRMSProp(
                var=parameter.handle,
                mg=mg.handle,
                ms=ms.handle,
                mom=mom.handle,
                lr=learning_rate,
                rho=decay,
                momentum=momentum,
                epsilon=epsilon,
                grad=update,
                indices=indices)
          else:
            tf.raw_ops.ResourceSparseApplyRMSProp(
                var=parameter.handle,
                ms=ms.handle,
                mom=mom.handle,
                lr=learning_rate,
                rho=decay,
                momentum=momentum,
                epsilon=epsilon,
                grad=update,
                indices=indices)
        else:
          if self.centered:
            tf.raw_ops.ResourceApplyCenteredRMSProp(
                var=parameter.handle,
                mg=mg.handle,
                ms=ms.handle,
                mom=mom.handle,
                lr=learning_rate,
                rho=decay,
                momentum=momentum,
                epsilon=epsilon,
                grad=update)
          else:
            tf.raw_ops.ResourceApplyRMSProp(
                var=parameter.handle,
                ms=ms.handle,
                mom=mom.handle,
                lr=learning_rate,
                rho=decay,
                momentum=momentum,
                epsilon=epsilon,
                grad=update)
