# Copyright 2017 The Sonnet Authors. All Rights Reserved.
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

"""Layer normalization module for Sonnet.

This contains the module LayerNorm, which performs layer normalization on
its inputs.

Original paper: https://arxiv.org/abs/1607.06450.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from sonnet.python.modules import base
from sonnet.python.modules import util

import tensorflow as tf


class LayerNorm(base.AbstractModule):
  """Layer normalization module.

  Implementation based on:
  https://arxiv.org/abs/1607.06450

  This module transforms input x into:

    outputs = gamma * (x - mu) / sigma + beta

  where mu and sigma are respectively the mean and standard deviation of x.
  Gamma and beta are trainable parameters for scaling and shifting respectively.

  """

  GAMMA = "gamma"  # Layer norm scaling.
  BETA = "beta"  # Layer norm bias.

  POSSIBLE_INITIALIZER_KEYS = {GAMMA, BETA}
  # Keep old name for backwards compatibility

  POSSIBLE_KEYS = POSSIBLE_INITIALIZER_KEYS

  def __init__(self,
               eps=1e-5,
               initializers=None,
               partitioners=None,
               regularizers=None,
               name="layer_norm"):
    """Constructs a LayerNorm module.

    Args:
      eps: small epsilon to avoid division by zero variance. Defaults to
        1e-5 as used in the paper.
      initializers: Dict containing ops to initialize the scale
        (with key 'gamma') and bias (with key 'beta').
      partitioners: Optional dict containing partitioners to partition
        the scale (with key 'gamma') and bias (with key 'beta'). As a default,
        no partitioners are used.
      regularizers: Optional dict containing regularizers for the scale (with
        key 'gamma') and bias (with key 'beta').. As a default, no regularizers
        are used.
      name: name of the module.

    Raises:
      KeyError: If `initializers`, `partitioners` or `regularizers` contain
        any keys other than `gamma` or `beta`.
      TypeError: If any of the given initializers, partitioners or regularizers
        are not callable.
    """
    super(LayerNorm, self).__init__(name=name)

    self._eps = eps

    self._initializers = util.check_initializers(initializers,
                                                 self.POSSIBLE_INITIALIZER_KEYS)
    self._partitioners = util.check_partitioners(partitioners,
                                                 self.POSSIBLE_INITIALIZER_KEYS)
    self._regularizers = util.check_regularizers(regularizers,
                                                 self.POSSIBLE_INITIALIZER_KEYS)

  def _build(self, inputs):
    """Connects the LayerNorm module into the graph.

    Args:
      inputs: a Tensor of shape `[batch_size, layer_dim]`.

    Returns:
      normalized: layer normalized outputs with same shape as inputs.

    Raises:
      base.NotSupportedError: If `inputs` has data type of `tf.float16` or
          `tf.bfloat16`.
    """

    if inputs.dtype in [tf.float16, tf.bfloat16]:
      raise base.NotSupportedError(
          "LayerNorm does not support `tf.float16` or `tf.bfloat16`, "
          "insufficient precision for calculating sufficient statistics.")

    if inputs.get_shape().ndims != 2:
      raise base.NotSupportedError(
          "Layer normalization expects inputs of rank 2."
          " Got inputs of rank {}.".format(inputs.get_shape().ndims))

    hidden_size = inputs.get_shape()[1].value

    if self.GAMMA not in self._initializers:
      self._initializers[self.GAMMA] = create_gamma_initializer()
    self._gamma = tf.get_variable(
        self.GAMMA,
        shape=[hidden_size],
        dtype=inputs.dtype,
        initializer=self._initializers[self.GAMMA],
        partitioner=self._partitioners.get(self.GAMMA),
        regularizer=self._regularizers.get(self.GAMMA))

    if self.BETA not in self._initializers:
      self._initializers[self.BETA] = create_beta_initializer()
    self._beta = tf.get_variable(
        self.BETA,
        shape=[hidden_size],
        dtype=inputs.dtype,
        initializer=self._initializers[self.BETA],
        partitioner=self._partitioners.get(self.BETA),
        regularizer=self._regularizers.get(self.BETA))

    mean, var = tf.nn.moments(inputs, [1], keep_dims=True)

    normalized = tf.nn.batch_normalization(inputs, mean, var, self._beta,
                                           self._gamma, self._eps)
    return normalized

  @property
  def initializers(self):
    return self._initializers

  @property
  def partitioners(self):
    return self._partitioners

  @property
  def regularizers(self):
    return self._regularizers

  @property
  def beta(self):
    self._ensure_is_connected()
    return self._beta

  @property
  def gamma(self):
    self._ensure_is_connected()
    return self._gamma


def create_beta_initializer():
  """Returns a default initializer for the `beta` in layer norm."""
  return tf.zeros_initializer()


def create_gamma_initializer():
  """Returns a default initializer for the `gamma` in layer norm."""
  return tf.ones_initializer()
