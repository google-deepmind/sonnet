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


class Adam(base.Optimizer):
  """Adaptive Moment Estimation (Adam) module.

  https://arxiv.org/abs/1412.6980

  Attributes:
    learning_rate: Learning rate.
    beta1: Beta1.
    beta2: Beta2.
    epsilon: Small value to avoid zero denominator.
    step: Step count.
    m: Accumulated m for each parameter.
    v: Accumulated v for each parameter.
  """

  def __init__(self,
               learning_rate,  # TODO(petebu): Consider a default here.
               beta1=0.9,
               beta2=0.999,
               epsilon=1e-8,
               name=None):
    """Constructs an `Adam` module.

    Args:
      learning_rate: Learning rate.
      beta1: Beta1.
      beta2: Beta2.
      epsilon: Small value to avoid zero denominator.
      name: Name of the module.
    """
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
    zero_var = lambda p: utils.variable_like(p, trainable=False)
    with tf.name_scope("m"):
      self.m.extend(zero_var(p) for p in parameters)
    with tf.name_scope("v"):
      self.v.extend(zero_var(p) for p in parameters)

  def apply(self, updates, parameters):
    r"""Applies updates to parameters.

    Applies the Adam update rule for each update, parameter pair:

        m_t <- beta1 * m_{t-1} + (1 - beta1) * update
        v_t <- beta2 * v_{t-1} + (1 - beta2) * update * update

        \hat{m}_t <- m_t / (1 - beta1^t)
        \hat{v}_t <- v_t / (1 - beta2^t)
        scaled_update <- \hat{m}_t / (sqrt(\hat{v}_t) + epsilon)

        parameter <- parameter - learning_rate * scaled_update

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
    self.step.assign_add(1)
    for update, parameter, m, v in zip(updates, parameters, self.m, self.v):
      if update is not None:
        optimizer_utils.check_same_dtype(update, parameter)
        lr = tf.cast(self.learning_rate, update.dtype)
        beta1 = tf.cast(self.beta1, update.dtype)
        beta2 = tf.cast(self.beta2, update.dtype)
        epsilon = tf.cast(self.epsilon, update.dtype)
        step = tf.cast(self.step, update.dtype)
        if isinstance(update, tf.IndexedSlices):
          update, indices = optimizer_utils.deduplicate_indexed_slices(
              update.values, update.indices)
          sparse_m_update = (beta1 * m.sparse_read(indices) +
                             (1. - beta1) * update)
          m.scatter_update(tf.IndexedSlices(sparse_m_update, indices))
          sparse_v_update = (beta2 * v.sparse_read(indices) +
                             (1. - beta2) * tf.square(update))
          v.scatter_update(tf.IndexedSlices(sparse_v_update, indices))
          debiased_m = sparse_m_update / (1. - tf.pow(beta1, step))
          debiased_v = sparse_v_update / (1. - tf.pow(beta2, step))
          adam_update = lr * debiased_m / (tf.sqrt(debiased_v) + epsilon)
          parameter.scatter_sub(tf.IndexedSlices(adam_update, indices))
        else:
          m.assign((beta1 * m) + (1. - beta1) * update)
          v.assign((beta2 * v) + (1. - beta2) * tf.square(update))
          debiased_m = m / (1. - tf.pow(beta1, step))
          debiased_v = v / (1. - tf.pow(beta2, step))
          adam_update = lr * debiased_m / (tf.sqrt(debiased_v) + epsilon)
          parameter.assign_sub(adam_update)


class FastAdam(base.Optimizer):
  """Adaptive Moment Estimation (Adam) module."""

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
    zero_var = lambda p: utils.variable_like(p, trainable=False)
    with tf.name_scope("m"):
      self.m.extend(zero_var(p) for p in parameters)
    with tf.name_scope("v"):
      self.v.extend(zero_var(p) for p in parameters)

  def apply(self, updates, parameters):
    """Applies updates to parameters."""
    optimizer_utils.check_distribution_strategy()
    optimizer_utils.check_updates_parameters(updates, parameters)
    self._initialize(parameters)
    self.step.assign_add(1)
    for update, parameter, m, v in zip(updates, parameters, self.m, self.v):
      if update is not None:
        optimizer_utils.check_same_dtype(update, parameter)
        lr = tf.cast(self.learning_rate, update.dtype)
        beta1 = tf.cast(self.beta1, update.dtype)
        beta2 = tf.cast(self.beta2, update.dtype)
        epsilon = tf.cast(self.epsilon, update.dtype)
        step = tf.cast(self.step, update.dtype)
        if isinstance(update, tf.IndexedSlices):
          update, indices = optimizer_utils.deduplicate_indexed_slices(
              update.values, update.indices)
          sparse_m_update = (beta1 * m.sparse_read(indices) +
                             (1. - beta1) * update)
          m.scatter_update(tf.IndexedSlices(sparse_m_update, indices))
          sparse_v_update = (beta2 * v.sparse_read(indices) +
                             (1. - beta2) * tf.square(update))
          v.scatter_update(tf.IndexedSlices(sparse_v_update, indices))
          debiased_m = sparse_m_update / (1. - tf.pow(beta1, step))
          debiased_v = sparse_v_update / (1. - tf.pow(beta2, step))
          adam_update = lr * debiased_m / (tf.sqrt(debiased_v) + epsilon)
          parameter.scatter_sub(tf.IndexedSlices(adam_update, indices))
        else:
          beta1_power = tf.pow(beta1, step)
          beta2_power = tf.pow(beta2, step)
          tf.raw_ops.ResourceApplyAdam(
              var=parameter.handle,
              m=m.handle,
              v=v.handle,
              beta1_power=beta1_power,
              beta2_power=beta2_power,
              lr=lr,
              beta1=beta1,
              beta2=beta2,
              epsilon=epsilon,
              grad=update)
