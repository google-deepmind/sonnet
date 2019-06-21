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

"""Adaptive Moment Estimation (Adam) module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sonnet.src import base
from sonnet.src import once
from sonnet.src import optimizer_utils
from sonnet.src import utils
import tensorflow as tf


class Adam(base.Module):
  """Adaptive Moment Estimation (Adam) module.

  https://arxiv.org/abs/1412.6980
  """

  def __init__(self,
               learning_rate,  # TODO(petebu): Consider a default here.
               beta1=0.9,
               beta2=0.999,
               epsilon=1e-8,
               name=None):
    """Constructs an `Adam` module."""
    super(Adam, self).__init__(name)
    self.learning_rate = learning_rate
    self.beta1 = beta1
    self.beta2 = beta2
    self.epsilon = epsilon
    # TODO(petebu): Consider allowing the user to pass in a step.
    self.step = tf.Variable(0, trainable=False, name="t", dtype=tf.int64)
    self.m = []
    self.v = []

  @once.once
  def _initialize(self, parameters):
    optimizer_utils.check_strategy()
    zero_var = lambda p: utils.variable_like(p, trainable=False)
    with tf.name_scope("m"):
      self.m.extend(zero_var(p) for p in parameters)
    with tf.name_scope("v"):
      self.v.extend(zero_var(p) for p in parameters)

  def apply(self, updates, parameters):
    """Apply updates to parameters.

    Applies the Adam update rule for each update, parameter pair:

        alpha <- learning_rate * sqrt(1 - beta2^t) / (1 - beta1^t)
        m_t <- beta1 * m_{t-1} + (1 - beta1) * update
        v_t <- beta2 * v_{t-1} + (1 - beta2) * update * update
        parameter <- parameter - alpha * m_t / (sqrt(v_t) + epsilon)

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
    self.step.assign_add(1)
    for update, parameter, m, v in zip(updates, parameters, self.m, self.v):
      # TODO(petebu): Add support for sparse tensors.
      # TODO(petebu): Consider caching learning_rate cast.
      # TODO(petebu): Consider the case when all updates are None.
      if update is not None:
        optimizer_utils.check_same_dtype(update, parameter)
        learning_rate = tf.cast(self.learning_rate, update.dtype.base_dtype)
        beta1 = tf.cast(self.beta1, update.dtype.base_dtype)
        beta2 = tf.cast(self.beta2, update.dtype.base_dtype)
        epsilon = tf.cast(self.epsilon, update.dtype.base_dtype)
        step = tf.cast(self.step, update.dtype.base_dtype)

        inv_beta1 = 1. - beta1
        inv_beta2 = 1. - beta2
        inv_beta1_power = 1. - tf.pow(beta1, step)
        inv_beta2_power = 1. - tf.pow(beta2, step)
        m.assign((beta1 * m) + inv_beta1 * update)
        v.assign((beta2 * v) + inv_beta2 * tf.square(update))
        adam_update = learning_rate * (
            (m / inv_beta1_power) / (tf.sqrt(v / inv_beta2_power) + epsilon))
        parameter.assign_sub(adam_update)


class FastAdam(base.Module):
  """Faster Adaptive Moment Estimation (Adam) module."""

  def __init__(self,
               learning_rate,  # TODO(petebu): Consider a default here.
               beta1=0.9,
               beta2=0.999,
               epsilon=1e-8,
               name=None):
    """Constructs an `Adam` module."""
    super(FastAdam, self).__init__(name)
    self.learning_rate = learning_rate
    self.beta1 = beta1
    self.beta2 = beta2
    self.epsilon = epsilon
    self.step = tf.Variable(0, trainable=False, name="t", dtype=tf.int64)
    self.m = []
    self.v = []

  @once.once
  def _initialize(self, parameters):
    optimizer_utils.check_strategy()
    zero_var = lambda p: utils.variable_like(p, trainable=False)
    with tf.name_scope("m"):
      self.m.extend(zero_var(p) for p in parameters)
    with tf.name_scope("v"):
      self.v.extend(zero_var(p) for p in parameters)

  def apply(self, updates, parameters):
    """Apply updates to parameters.

    Applies the Adam update rule for each update, parameter pair:

        alpha <- learning_rate * sqrt(1 - beta2^t) / (1 - beta1^t)
        m_t <- beta1 * m_{t-1} + (1 - beta1) * update
        v_t <- beta2 * v_{t-1} + (1 - beta2) * update * update
        parameter <- parameter - alpha * m_t / (sqrt(v_t) + epsilon)

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
    self.step.assign_add(1)
    for update, parameter, m, v in zip(updates, parameters, self.m, self.v):
      # TODO(petebu): Add support for sparse tensors.
      # TODO(petebu): Consider caching learning_rate cast.
      # TODO(petebu): Consider the case when all updates are None.
      if update is not None:
        optimizer_utils.check_same_dtype(update, parameter)
        learning_rate = tf.cast(self.learning_rate, update.dtype.base_dtype)
        beta1 = tf.cast(self.beta1, update.dtype.base_dtype)
        beta2 = tf.cast(self.beta2, update.dtype.base_dtype)
        epsilon = tf.cast(self.epsilon, update.dtype.base_dtype)
        step = tf.cast(self.step, update.dtype.base_dtype)

        beta1_power = tf.pow(beta1, step)
        beta2_power = tf.pow(beta2, step)
        tf.raw_ops.ResourceApplyAdam(
            var=parameter.handle,
            m=m.handle,
            v=v.handle,
            beta1_power=beta1_power,
            beta2_power=beta2_power,
            lr=learning_rate,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            grad=update)
