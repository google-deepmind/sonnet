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

from typing import Optional, Sequence, Union

from sonnet.src import base
from sonnet.src import once
from sonnet.src import types
from sonnet.src import utils
from sonnet.src.optimizers import optimizer_utils
import tensorflow as tf


def momentum_update(update, learning_rate, mu, momentum, use_nesterov):
  """Computes a momentum update for a single parameter."""
  momentum = (mu * momentum) + update
  if use_nesterov:
    update = learning_rate * ((mu * momentum) + update)
  else:
    update = learning_rate * momentum
  return update, momentum


class Momentum(base.Optimizer):
  """SGD with Momentum module.

  Attributes:
    learning_rate: Learning rate.
    momentum: Momentum scalar.
    use_nesterov: `True` if using Nesterov momentum.
    accumulated_momentum: Accumulated momentum for each parameter.
  """

  def __init__(self,
               learning_rate: Union[types.FloatLike, tf.Variable],
               momentum: Union[types.FloatLike, tf.Variable],
               use_nesterov: bool = False,
               name: Optional[str] = None):
    """Constructs a `Momentum` module.

    Args:
      learning_rate: Learning rate.
      momentum: Momentum scalar.
      use_nesterov: Whether to use Nesterov momentum.
      name: Name of the module.
    """
    super().__init__(name)
    self.learning_rate = learning_rate
    self.momentum = momentum  # TODO(petebu) Reconsider name.
    self.use_nesterov = use_nesterov
    self.accumulated_momentum = []  # TODO(petebu) Reconsider name.

  @once.once
  def _initialize(self, parameters):
    with tf.name_scope("accumulated_momentum"):
      self.accumulated_momentum.extend(
          utils.variable_like(p, trainable=False) for p in parameters)

  def apply(self, updates: Sequence[types.ParameterUpdate],
            parameters: Sequence[tf.Variable]):
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
      updates: A list of updates to apply to parameters. Updates are often
        gradients as returned by `tf.GradientTape.gradient`.
      parameters: A list of parameters. A parameter is a `tf.Variable`.

    Raises:
      ValueError: If `updates` and `parameters` are empty, have different
        lengths, or have inconsistent types.
    """
    optimizer_utils.check_distribution_strategy()
    optimizer_utils.check_updates_parameters(updates, parameters)
    self._initialize(parameters)
    for update, param, momentum_var in zip(updates, parameters,
                                           self.accumulated_momentum):
      if update is None:
        continue

      optimizer_utils.check_same_dtype(update, param)
      learning_rate = tf.cast(self.learning_rate, update.dtype)
      mu = tf.cast(self.momentum, update.dtype)

      if isinstance(update, tf.IndexedSlices):
        # Sparse read our state.
        update, indices = optimizer_utils.deduplicate_indexed_slices(update)
        momentum = momentum_var.sparse_read(indices)

        # Compute and apply a sparse update to our parameter and state.
        update, momentum = momentum_update(update, learning_rate, mu, momentum,
                                           self.use_nesterov)
        momentum_var.scatter_update(tf.IndexedSlices(momentum, indices))
        param.scatter_sub(tf.IndexedSlices(update, indices))

      else:
        # Compute and apply a dense update.
        update, momentum = momentum_update(update, learning_rate, mu,
                                           momentum_var, self.use_nesterov)
        momentum_var.assign(momentum)
        param.assign_sub(update)
