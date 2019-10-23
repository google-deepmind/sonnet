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
# from __future__ import google_type_annotations
from __future__ import print_function

from sonnet.src import base
from sonnet.src import once
from sonnet.src import types
from sonnet.src import utils
from sonnet.src.optimizers import optimizer_utils

import tensorflow as tf
from typing import Optional, Sequence, Text, Union


def adam_update(update, learning_rate, beta1, beta2, epsilon, step, m, v):
  """Computes the 'ADAM' update for a single parameter."""
  m = beta1 * m + (1. - beta1) * update
  v = beta2 * v + (1. - beta2) * tf.square(update)
  debiased_m = m / (1. - tf.pow(beta1, step))
  debiased_v = v / (1. - tf.pow(beta2, step))
  update = learning_rate * debiased_m / (tf.sqrt(debiased_v) + epsilon)
  return update, m, v


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

  def __init__(
      self,
      # TODO(petebu): Consider a default learning rate.
      learning_rate: Union[types.FloatLike, tf.Variable],
      beta1: Union[types.FloatLike, tf.Variable] = 0.9,
      beta2: Union[types.FloatLike, tf.Variable] = 0.999,
      epsilon: Union[types.FloatLike, tf.Variable] = 1e-8,
      name: Optional[Text] = None):
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
  def _initialize(self, parameters: Sequence[tf.Variable]):
    zero_var = lambda p: utils.variable_like(p, trainable=False)
    with tf.name_scope("m"):
      self.m.extend(zero_var(p) for p in parameters)
    with tf.name_scope("v"):
      self.v.extend(zero_var(p) for p in parameters)

  def apply(self, updates: Sequence[types.ParameterUpdate],
            parameters: Sequence[tf.Variable]):
    r"""Applies updates to parameters.

    Applies the Adam update rule for each update, parameter pair:

        m_t <- beta1 * m_{t-1} + (1 - beta1) * update
        v_t <- beta2 * v_{t-1} + (1 - beta2) * update * update

        \hat{m}_t <- m_t / (1 - beta1^t)
        \hat{v}_t <- v_t / (1 - beta2^t)
        scaled_update <- \hat{m}_t / (sqrt(\hat{v}_t) + epsilon)

        parameter <- parameter - learning_rate * scaled_update

    Args:
      updates: A list of updates to apply to parameters. Updates are often
        gradients as returned by `tf.GradientTape.gradient`.
      parameters: A list of parameters.

    Raises:
      ValueError: If `updates` and `parameters` are empty, have different
        lengths, or have inconsistent types.
    """
    optimizer_utils.check_distribution_strategy()
    optimizer_utils.check_updates_parameters(updates, parameters)
    self._initialize(parameters)
    self.step.assign_add(1)
    for update, param, m_var, v_var in zip(updates, parameters, self.m, self.v):
      if update is None:
        continue

      optimizer_utils.check_same_dtype(update, param)
      learning_rate = tf.cast(self.learning_rate, update.dtype)
      beta1 = tf.cast(self.beta1, update.dtype)
      beta2 = tf.cast(self.beta2, update.dtype)
      epsilon = tf.cast(self.epsilon, update.dtype)
      step = tf.cast(self.step, update.dtype)

      if isinstance(update, tf.IndexedSlices):
        # Sparse read our state.
        update, indices = optimizer_utils.deduplicate_indexed_slices(update)
        m = m_var.sparse_read(indices)
        v = v_var.sparse_read(indices)

        # Compute and apply a sparse update to our parameter and state.
        update, m, v = adam_update(update, learning_rate, beta1, beta2, epsilon,
                                   step, m, v)
        param.scatter_sub(tf.IndexedSlices(update, indices))
        m_var.scatter_update(tf.IndexedSlices(m, indices))
        v_var.scatter_update(tf.IndexedSlices(v, indices))

      else:
        # Compute and apply a dense update to our parameter and state.
        update, m, v = adam_update(update, learning_rate, beta1, beta2, epsilon,
                                   step, m_var, v_var)
        param.assign_sub(update)
        m_var.assign(m)
        v_var.assign(v)
