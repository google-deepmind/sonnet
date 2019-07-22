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

"""Stochastic Gradient Descent module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sonnet.src import base
from sonnet.src import optimizer_utils
import tensorflow as tf


class SGD(base.Optimizer):
  """Stochastic Gradient Descent (SGD) module.

  Attributes:
    learning_rate: Learning rate.
  """

  def __init__(self, learning_rate, name=None):
    """Constructs an `SGD` module.

    Args:
      learning_rate: Learning rate.
      name: Name of the module.
    """
    super(SGD, self).__init__(name)
    self.learning_rate = learning_rate

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
    for update, parameter in zip(updates, parameters):
      if update is not None:
        optimizer_utils.check_same_dtype(update, parameter)
        learning_rate = tf.cast(self.learning_rate, update.dtype)
        if isinstance(update, tf.IndexedSlices):
          parameter.scatter_sub(
              tf.IndexedSlices(update.values * learning_rate, update.indices))
        else:
          parameter.assign_sub(update * learning_rate)


class FastSGD(base.Optimizer):
  """Stochastic Gradient Descent (SGD) module."""

  def __init__(self, learning_rate, name=None):
    """Constructs an `SGD` module."""
    super(FastSGD, self).__init__(name)
    self.learning_rate = learning_rate

  def apply(self, updates, parameters):
    """Applies updates to parameters."""
    optimizer_utils.check_distribution_strategy()
    optimizer_utils.check_updates_parameters(updates, parameters)
    for update, parameter in zip(updates, parameters):
      if update is not None:
        optimizer_utils.check_same_dtype(update, parameter)
        learning_rate = tf.cast(self.learning_rate, update.dtype)
        if isinstance(update, tf.IndexedSlices):
          parameter.scatter_nd_sub(
              update.indices, update.values * learning_rate)
        else:
          tf.raw_ops.ResourceApplyGradientDescent(
              var=parameter.handle,
              alpha=learning_rate,
              delta=update)
