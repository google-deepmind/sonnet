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

"""Linear module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from sonnet.src import base
from sonnet.src import initializers
from sonnet.src import once
from sonnet.src import utils
import tensorflow as tf


class Linear(base.Module):
  """Linear module, optionally including bias."""

  def __init__(self,
               output_size,
               with_bias=True,
               w_init=None,
               b_init=None,
               name=None):
    """Constructs a `Linear` module.

    Args:
      output_size: Output dimensionality.
      with_bias: Whether to include bias parameters. Default `True`.
      w_init: Optional initializer for the weights. By default the weights are
        initialized truncated random normal values with a standard deviation of
        `1 / sqrt(input_feature_size)`, which is commonly used when the inputs
        are zero centered (see https://arxiv.org/abs/1502.03167v3).
      b_init: Optional initializer for the bias. By default the bias is
        initialized to zero.
      name: Name of the module.
    """
    super(Linear, self).__init__(name=name)
    self.output_size = output_size
    self.with_bias = with_bias
    self.w_init = w_init
    if with_bias:
      self.b_init = b_init if b_init is not None else initializers.Zeros()
    elif b_init is not None:
      raise ValueError("When not using a bias the b_init must be None.")

  @once.once
  def _initialize(self, inputs):
    """Constructs parameters used by this module."""
    utils.assert_rank(inputs, 2)

    input_size = inputs.shape[1]
    if input_size is None:  # Can happen inside an @tf.function.
      raise ValueError("Input size must be specified at module build time.")

    self.input_size = input_size

    if self.w_init is None:
      # See https://arxiv.org/abs/1502.03167v3.
      stddev = 1 / math.sqrt(self.input_size)
      self.w_init = initializers.TruncatedNormal(stddev=stddev)

    self.w = tf.Variable(
        self.w_init([self.input_size, self.output_size], inputs.dtype),
        name="w")

    if self.with_bias:
      self.b = tf.Variable(self.b_init([self.output_size], inputs.dtype),
                           name="b")

  def __call__(self, inputs):
    self._initialize(inputs)

    outputs = tf.matmul(inputs, self.w)
    if self.with_bias:
      outputs = tf.add(outputs, self.b)
    return outputs
