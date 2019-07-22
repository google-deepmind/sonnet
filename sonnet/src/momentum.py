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

"""SGD with Momentum module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sonnet.src import base
from sonnet.src import once
from sonnet.src import optimizer_utils
from sonnet.src import utils
import tensorflow as tf


class Momentum(base.Optimizer):
  """SGD with Momentum module.

  Attributes:
    learning_rate: Learning rate.
    momentum: Momentum scalar.
    use_nesterov: `True` if using Nesterov momentum.
    accumulated_momentum: Accumulated momentum for each parameter.
  """

  def __init__(self, learning_rate, momentum, use_nesterov=False, name=None):
    """Constructs a `Momentum` module.

    Args:
      learning_rate: Learning rate.
      momentum: Momentum scalar.
      use_nesterov: Whether to use Nesterov momentum.
      name: Name of the module.
    """
    super(Momentum, self).__init__(name)
    self.learning_rate = learning_rate
    self.momentum = momentum  # TODO(petebu) Reconsider name.
    self.use_nesterov = use_nesterov
    self.accumulated_momentum = []  # TODO(petebu) Reconsider name.

  @once.once
  def _initialize(self, parameters):
    with tf.name_scope("accumulated_momentum"):
      self.accumulated_momentum.extend(
          utils.variable_like(p, trainable=False) for p in parameters)

  def apply(self, updates, parameters):
    """Applies updates to parameters.

    By default it applies the momentum update rule for each update, parameter
    pair:

        accum_t <- momentum * accum_{t-1} + update
        parameter <- parameter - learning_rate * accum_t

    And when using Nesterov momentum (`use_nesterov=True`) it applies:

        accum_t <- momentum * accum_{t-1} + update
        parameter <- parameter - (learning_rate * update +
                                  learning_rate * momentum * accum_t)

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
    for update, parameter, momentum in zip(
        updates, parameters, self.accumulated_momentum):
      if update is not None:
        optimizer_utils.check_same_dtype(update, parameter)
        lr = tf.cast(self.learning_rate, update.dtype)
        mu = tf.cast(self.momentum, update.dtype)
        if isinstance(update, tf.IndexedSlices):
          update, indices = optimizer_utils.deduplicate_indexed_slices(
              update.values, update.indices)
          sparse_momentum_update = (mu * momentum.sparse_read(indices)) + update
          momentum.scatter_update(
              tf.IndexedSlices(sparse_momentum_update, indices))
          if self.use_nesterov:
            parameter.scatter_sub(tf.IndexedSlices(
                (lr * update) + (lr * mu * sparse_momentum_update), indices))
          else:
            parameter.scatter_sub(tf.IndexedSlices(
                lr * sparse_momentum_update, indices))
        else:
          momentum.assign((mu * momentum) + update)
          if self.use_nesterov:
            parameter.assign_sub((lr * update) + (lr * mu * momentum))
          else:
            parameter.assign_sub(lr * momentum)


class FastMomentum(base.Optimizer):
  """SGD with Momentum module."""

  def __init__(self, learning_rate, momentum, use_nesterov=False, name=None):
    """Constructs a `Momentum` module."""
    super(FastMomentum, self).__init__(name)
    self.learning_rate = learning_rate
    self.momentum = momentum
    self.use_nesterov = use_nesterov
    self.accumulated_momentum = []

  @once.once
  def _initialize(self, parameters):
    with tf.name_scope("accumulated_momentum"):
      self.accumulated_momentum.extend(
          utils.variable_like(p, trainable=False) for p in parameters)

  def apply(self, updates, parameters):
    """Applies updates to parameters."""
    optimizer_utils.check_distribution_strategy()
    optimizer_utils.check_updates_parameters(updates, parameters)
    self._initialize(parameters)
    for update, parameter, accumulated_momentum in zip(
        updates, parameters, self.accumulated_momentum):
      if update is not None:
        optimizer_utils.check_same_dtype(update, parameter)
        learning_rate = tf.cast(self.learning_rate, update.dtype)
        momentum = tf.cast(self.momentum, update.dtype)
        if isinstance(update, tf.IndexedSlices):
          update, indices = optimizer_utils.deduplicate_indexed_slices(
              update.values, update.indices)
          tf.raw_ops.ResourceSparseApplyMomentum(
              var=parameter.handle,
              accum=accumulated_momentum.handle,
              lr=learning_rate,
              grad=update,
              indices=indices,
              momentum=momentum,
              use_nesterov=self.use_nesterov)
        else:
          tf.raw_ops.ResourceApplyMomentum(
              var=parameter.handle,
              accum=accumulated_momentum.handle,
              lr=learning_rate,
              grad=update,
              momentum=momentum,
              use_nesterov=self.use_nesterov)
