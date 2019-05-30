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
from sonnet.src import optimizer_utils
import tensorflow as tf


class Momentum(base.Module):
  """SGD with Momentum module.

  By default it applies the momentum update rule for each update, parameter
  pair:

      accum_t <- momentum * accum_{t-1} + update
      parameter <- parameter - learning_rate * accum_t

  And when using Nesterov momentum (`use_nesterov=True`) it applies:

      accum_t <- momentum * accum_{t-1} + update
      parameter <- parameter - (learning_rate * update +
                                learning_rate * momentum * accum_t)
  """

  def __init__(self, learning_rate, momentum, use_nesterov=False, name=None):
    """Constructs a `Momentum` module."""
    super(Momentum, self).__init__(name)
    self.learning_rate = learning_rate
    self.momentum = momentum
    self.use_nesterov = use_nesterov
    self.accumulated_momentum = {}

  def _get_accumulated_momentum(self, variable):
    # TODO(petebu): Consider using a checkpointable dict.
    accum = self.accumulated_momentum.get(variable, None)
    if accum is None:
      accum_name = "accum/" + variable.name.replace(":0", "")
      with tf.device(variable.device):
        # TODO(petebu): Consider setting the dtype to equal that of variable.
        accum = tf.Variable(
            tf.zeros_like(variable), name=accum_name, trainable=False)
      self.accumulated_momentum[variable] = accum
    return accum

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
        accumulated_momentum = self._get_accumulated_momentum(parameter)
        learning_rate = tf.cast(self.learning_rate, update.dtype.base_dtype)
        momentum = tf.cast(self.momentum, update.dtype.base_dtype)
        tf.raw_ops.ResourceApplyMomentum(
            var=parameter.handle,
            accum=accumulated_momentum.handle,
            lr=learning_rate,
            grad=update,
            momentum=momentum,
            use_nesterov=self.use_nesterov)


class ReferenceMomentum(base.Module):
  """Reference version of the `Momentum` module.

  This is a reference implementation of the Momentum module. It doesn't use
  `tf.raw_ops` so it will be slower but you may find it easier to customize. It
  is fully tested and its behaviour matches the `tf.raw_ops` version. If you
  need a custom variant of `Momentum`, we recommend starting with this.

  By default it applies the momentum update rule for each update, parameter
  pair:

      accum_t <- momentum * accum_{t-1} + update
      parameter <- parameter - learning_rate * accum_t

  And when using Nesterov momentum (`use_nesterov=True`) it applies:

      accum_t <- momentum * accum_{t-1} + update
      parameter <- parameter - (learning_rate * update +
                                learning_rate * momentum * accum_t)
  """

  def __init__(self, learning_rate, momentum, use_nesterov=False, name=None):
    """Constructs a `ReferenceMomentum` module."""
    super(ReferenceMomentum, self).__init__(name)
    self.learning_rate = learning_rate
    self.momentum = momentum
    self.use_nesterov = use_nesterov
    self.accumulated_momentum = {}

  def _get_accumulated_momentum(self, variable):
    # TODO(petebu): Consider using a checkpointable dict.
    # TODO(petebu): Should we weakref the variables?
    accum = self.accumulated_momentum.get(variable, None)
    if accum is None:
      accum_name = "accum/" + variable.name.replace(":0", "")
      with tf.device(variable.device):
        # TODO(petebu): Consider setting the dtype to equal that of variable.
        accum = tf.Variable(
            tf.zeros_like(variable), name=accum_name, trainable=False)
      self.accumulated_momentum[variable] = accum
    return accum

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
        accumulated_momentum = self._get_accumulated_momentum(parameter)
        learning_rate = tf.cast(self.learning_rate, update.dtype.base_dtype)
        momentum = tf.cast(self.momentum, update.dtype.base_dtype)
        # TODO(petebu): Use a tf.CriticalSection for the assignments.
        accumulated_momentum.assign((momentum * accumulated_momentum) + update)
        if self.use_nesterov:
          parameter.assign_sub(learning_rate * update +
                               learning_rate * momentum * accumulated_momentum)
        else:
          parameter.assign_sub(learning_rate * accumulated_momentum)
