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

import itertools
from typing import Optional, Sequence, Union

from sonnet.src import base
from sonnet.src import once
from sonnet.src import types
from sonnet.src import utils
from sonnet.src.optimizers import optimizer_utils
import tensorflow as tf


def rmsprop_update(update, decay, learning_rate, epsilon, mu, mom, ms, mg):
  """Computes a single RMSProp update."""
  ms = tf.square(update) * (1. - decay) + ms * decay
  if mg is not None:  # centered
    mg = update * (1. - decay) + mg * decay
    denominator = ms - tf.square(mg) + epsilon
  else:
    denominator = ms + epsilon
  mom = (mu * mom) + (learning_rate * update * tf.math.rsqrt(denominator))
  return mom, ms, mg


class RMSProp(base.Optimizer):
  """RMSProp module.

  See: http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf

  Maintain a moving (discounted) average of the square of updates. Divides each
  update by the root of this average.

      ms <- decay * ms + (1-decay) * update^2
      mom <- momentum * mom + learning_rate * update / sqrt(ms + epsilon)
      parameter <- parameter - mom

  This implementation of `RMSprop` uses plain momentum, not Nesterov momentum.

  The centered version additionally maintains a moving average of the
  gradients, and uses that average to estimate the variance:

      mg <- decay * mg + (1-decay) * update
      ms <- decay * ms + (1-decay) * update^2
      mom <- momentum * mom + learning_rate * update / sqrt(ms - mg^2 + epsilon)
      parameter <- parameter - mom

  Attributes:
    learning_rate: Learning rate.
    decay: Learning rate decay over each update.
    momentum: Momentum scalar.
    epsilon: Small value to avoid zero denominator.
    centered: `True` if centered.
    mom: Accumulated mom for each parameter.
    ms: Accumulated ms for each parameter.
    mg: Accumulated mg for each parameter.
  """

  def __init__(self,
               learning_rate: Union[types.FloatLike, tf.Variable],
               decay: Union[types.FloatLike, tf.Variable] = 0.9,
               momentum: Union[types.FloatLike, tf.Variable] = 0.0,
               epsilon: Union[types.FloatLike, tf.Variable] = 1e-10,
               centered: bool = False,
               name: Optional[str] = None):
    """Constructs an `RMSProp` module.

    Args:
      learning_rate: Learning rate.
      decay: Learning rate decay over each update.
      momentum: Momentum scalar.
      epsilon: Small value to avoid zero denominator.
      centered: If True, gradients are normalized by the estimated variance of
        the gradient; if False, by the uncentered second moment. Setting this to
        True may help with training, but is slightly more expensive in terms of
        computation and memory. Defaults to False.
      name: Name for this module.
    """
    super().__init__(name)
    self.learning_rate = learning_rate
    self.decay = decay
    self.momentum = momentum
    self.epsilon = epsilon
    self.centered = centered
    self.mom = []
    self.ms = []
    self.mg = []

  @once.once
  def _initialize(self, parameters: Sequence[tf.Variable]):
    zero_var = lambda p: utils.variable_like(p, trainable=False)
    with tf.name_scope("momentum"):
      self.mom.extend(zero_var(p) for p in parameters)
    with tf.name_scope("rms"):
      self.ms.extend(zero_var(p) for p in parameters)
    if self.centered:
      with tf.name_scope("mg"):
        self.mg.extend(zero_var(p) for p in parameters)

  def apply(self, updates: Sequence[types.ParameterUpdate],
            parameters: Sequence[tf.Variable]):
    """Applies updates to parameters.

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
    for update, parameter, mom_var, ms_var, mg_var in itertools.zip_longest(
        updates, parameters, self.mom, self.ms, self.mg):
      if update is None:
        continue

      optimizer_utils.check_same_dtype(update, parameter)
      learning_rate = tf.cast(self.learning_rate, update.dtype)
      decay = tf.cast(self.decay, update.dtype)
      mu = tf.cast(self.momentum, update.dtype)
      epsilon = tf.cast(self.epsilon, update.dtype)

      if isinstance(update, tf.IndexedSlices):
        # Sparse read our state.
        update, indices = optimizer_utils.deduplicate_indexed_slices(update)
        ms = ms_var.sparse_read(indices)
        mg = mg_var.sparse_read(indices) if self.centered else None
        mom = mom_var.sparse_read(indices)

        # Compute and apply a sparse update to our parameter and state.
        mom, ms, mg = rmsprop_update(update, decay, learning_rate, epsilon, mu,
                                     mom, ms, mg)
        parameter.scatter_sub(tf.IndexedSlices(mom, indices))
        mom_var.scatter_update(tf.IndexedSlices(mom, indices))
        ms_var.scatter_update(tf.IndexedSlices(ms, indices))
        if self.centered:
          mg_var.scatter_update(tf.IndexedSlices(mg, indices))

      else:
        # Compute and apply a dense update to our parameters and state.
        mom, ms, mg = rmsprop_update(update, decay, learning_rate, epsilon, mu,
                                     mom_var, ms_var, mg_var)
        parameter.assign_sub(mom)
        mom_var.assign(mom)
        ms_var.assign(ms)
        if self.centered:
          mg_var.assign(mg)
