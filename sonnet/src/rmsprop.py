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

from sonnet.src import base
from sonnet.src import optimizer_utils
import tensorflow as tf


class RMSProp(base.Module):
  """RMSProp module.

  See: http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf

  Maintain a moving (discounted) average of the square of updates. Divides each
  update by the root of this average.

      ms <- decay * ms + (1-decay) * update^2
      mom = momentum * mom + learning_rate * update / sqrt(ms + epsilon)
      parameter := parameter - mom

  This implementation of `RMSprop` uses plain momentum, not Nesterov momentum.

  The centered version additionally maintains a moving average of the
  gradients, and uses that average to estimate the variance:

      mg = decay * mg + (1-decay) * update
      ms = decay * ms + (1-decay) * update^2
      mom = momentum * mom + learning_rate * update / sqrt(ms - mg^2 + epsilon)
      parameter := parameter - mom
  """

  def __init__(self, learning_rate, decay=0.9, momentum=0.0, epsilon=1e-10,
               centered=False, name=None):
    """Constructs an `RMSProp` module.

    Args:
      learning_rate: Learning rate.
      decay: Learning rate decay over each update.
      momentum: Momentum.
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
    self.moving_variables = {}  # TODO(petebu): Revisit this name.

  def _get_or_create_moving_vars(self, variable):
    # TODO(petebu): Consider using a checkpointable dict.
    mom, ms, mg = self.moving_variables.get(variable, (None, None, None))
    if mom is None:
      var_name = variable.name.replace(":0", "")
      with tf.device(variable.device):
        # TODO(petebu): Consider setting the dtype to equal that of variable.
        zeros = tf.zeros_like(variable)
        mom = tf.Variable(zeros, trainable=False, name="momentum/" + var_name)
        ms = tf.Variable(zeros, trainable=False, name="rms/" + var_name)
        if self.centered:
          mg = tf.Variable(zeros, trainable=False, name="mg/" + var_name)
      self.moving_variables[variable] = mom, ms, mg
    return mom, ms, mg

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
        mom, ms, mg = self._get_or_create_moving_vars(parameter)
        learning_rate = tf.cast(self.learning_rate, update.dtype.base_dtype)
        decay = tf.cast(self.decay, update.dtype.base_dtype)
        momentum = tf.cast(self.momentum, update.dtype.base_dtype)
        epsilon = tf.cast(self.epsilon, update.dtype.base_dtype)
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


class ReferenceRMSProp(base.Module):
  """Reference version of the RMSProp module.

  See: http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf

  Maintain a moving (discounted) average of the square of updates. Divides each
  update by the root of this average.

      ms <- decay * ms + (1-decay) * update^2
      mom = momentum * mom + learning_rate * update / sqrt(ms + epsilon)
      parameter := parameter - mom

  This implementation of RMSprop uses plain momentum, not Nesterov momentum.

  The centered version additionally maintains a moving average of the
  gradients, and uses that average to estimate the variance:

      mg = decay * mg + (1-decay) * update
      ms = decay * ms + (1-decay) * update^2
      mom = momentum * mom + learning_rate * update / sqrt(ms - mg^2 + epsilon)
      parameter := parameter - mom

  This is a reference implementation of the RMSProp module. It doesn't use
  raw_ops so it will be slower but you may find it easier to customize. It is
  fully tested and its behaviour matches the raw_ops version. If you need a
  custom variant of RMSProp, we recommend starting with this.
  """

  def __init__(self, learning_rate, decay=0.9, momentum=0.0, epsilon=1e-10,
               centered=False, name=None):
    """Constructs a reference RMSProp module.

    Args:
      learning_rate: Learning rate.
      decay: Learning rate decay over each update.
      momentum: Momentum.
      epsilon: Small value to avoid zero denominator.
      centered: If True, gradients are normalized by the estimated variance of
        the gradient; if False, by the uncentered second moment. Setting this to
        True may help with training, but is slightly more expensive in terms of
        computation and memory. Defaults to False.
      name: Name for this module.
    """
    super(ReferenceRMSProp, self).__init__(name)
    self.learning_rate = learning_rate
    self.decay = decay
    self.momentum = momentum
    self.epsilon = epsilon
    self.centered = centered
    self.moving_variables = {}

  def _get_or_create_moving_vars(self, variable):
    # TODO(petebu): Consider using a checkpointable dict.
    mom, ms, mg = self.moving_variables.get(variable, (None, None, None))
    if mom is None:
      var_name = variable.name.replace(":0", "")
      with tf.device(variable.device):
        # TODO(petebu): Consider setting the dtype to equal that of variable.
        zeros = tf.zeros_like(variable)
        mom = tf.Variable(zeros, trainable=False, name="momentum/" + var_name)
        ms = tf.Variable(zeros, trainable=False, name="rms/" + var_name)
        if self.centered:
          mg = tf.Variable(zeros, trainable=False, name="mg/" + var_name)
      self.moving_variables[variable] = mom, ms, mg
    return mom, ms, mg

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
        mom, ms, mg = self._get_or_create_moving_vars(parameter)
        learning_rate = tf.cast(self.learning_rate, update.dtype.base_dtype)
        decay = tf.cast(self.decay, update.dtype.base_dtype)
        momentum = tf.cast(self.momentum, update.dtype.base_dtype)
        epsilon = tf.cast(self.epsilon, update.dtype.base_dtype)

        # TODO(petebu): Use a tf.CriticalSection for the assignments.
        ms.assign(tf.square(update) * (1. - decay) + ms * decay)
        if self.centered:
          mg.assign(update * (1. - decay) + mg * decay)
          denominator = ms - mg + epsilon
        else:
          denominator = ms + epsilon
        mom.assign(momentum * mom + (
            learning_rate * update * tf.math.rsqrt(denominator)))
        parameter.assign_sub(mom)
