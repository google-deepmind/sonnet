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

from typing import Optional, Sequence, Union

from sonnet.src import base
from sonnet.src import once
from sonnet.src import types
from sonnet.src import utils
from sonnet.src.optimizers import optimizer_utils
import tensorflow as tf


def adam_update(g, alpha, beta_1, beta_2, epsilon, t, m, v):
  """Implements 'Algorithm 1' from :cite:`kingma2014adam`."""
  m = beta_1 * m + (1. - beta_1) * g      # Biased first moment estimate.
  v = beta_2 * v + (1. - beta_2) * g * g  # Biased second raw moment estimate.
  m_hat = m / (1. - tf.pow(beta_1, t))    # Bias corrected 1st moment estimate.
  v_hat = v / (1. - tf.pow(beta_2, t))    # Bias corrected 2nd moment estimate.
  update = alpha * m_hat / (tf.sqrt(v_hat) + epsilon)
  return update, m, v


class Adam(base.Optimizer):
  """Adaptive Moment Estimation (Adam) optimizer.

  Adam is an algorithm for first-order gradient-based optimization of stochastic
  objective functions, based on adaptive estimates of lower-order moments. See
  :cite:`kingma2014adam` for more details.

  Note: default parameter values have been taken from the paper.

  Attributes:
    learning_rate: Step size (``alpha`` in the paper).
    beta1: Exponential decay rate for first moment estimate.
    beta2: Exponential decay rate for second moment estimate.
    epsilon: Small value to avoid zero denominator.
    step: Step count.
    m: Biased first moment estimate (a list with one value per parameter).
    v: Biased second raw moment estimate (a list with one value per parameter).
  """

  def __init__(self,
               learning_rate: Union[types.FloatLike, tf.Variable] = 0.001,
               beta1: Union[types.FloatLike, tf.Variable] = 0.9,
               beta2: Union[types.FloatLike, tf.Variable] = 0.999,
               epsilon: Union[types.FloatLike, tf.Variable] = 1e-8,
               name: Optional[str] = None):
    """Constructs an `Adam` module.

    Args:
      learning_rate: Step size (``alpha`` in the paper).
      beta1: Exponential decay rate for first moment estimate.
      beta2: Exponential decay rate for second moment estimate.
      epsilon: Small value to avoid zero denominator.
      name: Name of the module.
    """
    super().__init__(name=name)
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
    """First and second order moments are initialized to zero."""
    zero_var = lambda p: utils.variable_like(p, trainable=False)
    with tf.name_scope("m"):
      self.m.extend(zero_var(p) for p in parameters)
    with tf.name_scope("v"):
      self.v.extend(zero_var(p) for p in parameters)

  def apply(self, updates: Sequence[types.ParameterUpdate],
            parameters: Sequence[tf.Variable]):
    r"""Applies updates to parameters.

    Applies the Adam update rule for each update, parameter pair:

    .. math::

       \begin{array}{ll}
       m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot update \\
       v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot update^2 \\
       \hat{m}_t = m_t / (1 - \beta_1^t) \\
       \hat{v}_t = v_t / (1 - \beta_2^t) \\
       delta = \alpha \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon) \\
       param_t = param_{t-1} - delta \\
       \end{array}

    Args:
      updates: A list of updates to apply to parameters. Updates are often
        gradients as returned by :tf:`GradientTape.gradient`.
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
      beta_1 = tf.cast(self.beta1, update.dtype)
      beta_2 = tf.cast(self.beta2, update.dtype)
      epsilon = tf.cast(self.epsilon, update.dtype)
      step = tf.cast(self.step, update.dtype)

      if isinstance(update, tf.IndexedSlices):
        # Sparse read our state.
        update, indices = optimizer_utils.deduplicate_indexed_slices(update)
        m = m_var.sparse_read(indices)
        v = v_var.sparse_read(indices)

        # Compute and apply a sparse update to our parameter and state.
        update, m, v = adam_update(
            g=update, alpha=learning_rate, beta_1=beta_1, beta_2=beta_2,
            epsilon=epsilon, t=step, m=m, v=v)
        param.scatter_sub(tf.IndexedSlices(update, indices))
        m_var.scatter_update(tf.IndexedSlices(m, indices))
        v_var.scatter_update(tf.IndexedSlices(v, indices))

      else:
        # Compute and apply a dense update to our parameter and state.
        update, m, v = adam_update(
            g=update, alpha=learning_rate, beta_1=beta_1, beta_2=beta_2,
            epsilon=epsilon, t=step, m=m_var, v=v_var)
        param.assign_sub(update)
        m_var.assign(m)
        v_var.assign(v)
