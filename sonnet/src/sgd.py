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


class SGD(base.Module):
  """Stochastic Gradient Descent module."""

  def __init__(self, learning_rate, name=None):
    """Constructs a Stochastic Gradient Descent module."""
    super(SGD, self).__init__(name)
    self.learning_rate = learning_rate

  def apply(self, updates, parameters):
    """Apply updates to parameters.

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
    for update, parameter in zip(updates, parameters):
      # TODO(petebu): Add support for sparse tensors.
      # TODO(petebu): Consider caching learning_rate cast.
      # TODO(petebu): Consider the case when all updates are None.
      if update is not None:
        optimizer_utils.check_same_dtype(update, parameter)
        tf.raw_ops.ResourceApplyGradientDescent(
            var=parameter.handle,
            alpha=tf.cast(self.learning_rate, update.dtype.base_dtype),
            delta=update)


class ReferenceSGD(base.Module):
  """Reference version of the Stochastic Gradient Descent module.

  This is a reference implementation of the SGD module. It doesn't use raw_ops
  so it will be slower but you may find it easier to customize. It is fully
  tested and its behaviour matches the raw_ops version. If you need a custom
  variant of SGD, we recommend starting with this.
  """

  def __init__(self, learning_rate, name=None):
    """Constructs a reference Stochastic Gradient Descent module."""
    super(ReferenceSGD, self).__init__(name)
    self.learning_rate = learning_rate

  def apply(self, updates, parameters):
    """Apply updates to parameters.

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
    for update, parameter in zip(updates, parameters):
      if update is not None:
        optimizer_utils.check_same_dtype(update, parameter)
        parameter.assign_sub(
            update * tf.cast(self.learning_rate, update.dtype.base_dtype))
