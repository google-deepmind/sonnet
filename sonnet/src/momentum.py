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


class Momentum(base.Module):
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
    self.momentum = momentum
    self.use_nesterov = use_nesterov
    self.accumulated_momentum = []

  @once.once
  def _initialize(self, parameters):
    optimizer_utils.check_strategy()
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
    optimizer_utils.check_updates_parameters(updates, parameters)
    self._initialize(parameters)
    for update, parameter, accumulated_momentum in zip(
        updates, parameters, self.accumulated_momentum):
      # TODO(petebu): Add support for sparse tensors.
      # TODO(petebu): Consider caching learning_rate cast.
      # TODO(petebu): Consider the case when all updates are None.
      if update is not None:
        optimizer_utils.check_same_dtype(update, parameter)
        learning_rate = tf.cast(self.learning_rate, update.dtype.base_dtype)
        momentum = tf.cast(self.momentum, update.dtype.base_dtype)

        accumulated_momentum.assign((momentum * accumulated_momentum) + update)
        if self.use_nesterov:
          parameter.assign_sub(learning_rate * update +
                               learning_rate * momentum * accumulated_momentum)
        else:
          parameter.assign_sub(learning_rate * accumulated_momentum)


class FastMomentum(base.Module):
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
    optimizer_utils.check_strategy()
    with tf.name_scope("accumulated_momentum"):
      self.accumulated_momentum.extend(
          utils.variable_like(p, trainable=False) for p in parameters)

  def apply(self, updates, parameters):
    """Applies updates to parameters."""
    optimizer_utils.check_updates_parameters(updates, parameters)
    self._initialize(parameters)
    for update, parameter, accumulated_momentum in zip(
        updates, parameters, self.accumulated_momentum):
      # TODO(petebu): Add support for sparse tensors.
      # TODO(petebu): Consider caching learning_rate cast.
      # TODO(petebu): Consider the case when all updates are None.
      if update is not None:
        optimizer_utils.check_same_dtype(update, parameter)
        learning_rate = tf.cast(self.learning_rate, update.dtype.base_dtype)
        momentum = tf.cast(self.momentum, update.dtype.base_dtype)

        tf.raw_ops.ResourceApplyMomentum(
            var=parameter.handle,
            accum=accumulated_momentum.handle,
            lr=learning_rate,
            grad=update,
            momentum=momentum,
            use_nesterov=self.use_nesterov)
