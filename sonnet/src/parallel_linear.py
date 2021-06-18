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
"""Parallel linear module."""

import math
from typing import Optional

from sonnet.src import base
from sonnet.src import initializers
from sonnet.src import once
from sonnet.src import utils
import tensorflow as tf


class ParallelLinears(base.Module):
  """Parallel linear.

  This is equivalent to n separate linears applied in parallel to n inputs. It
  takes an input of shape [num_linears, batch_size, input_size] and returns an
  output of shape [num_linears, batch_size, output_size].

  It uses a single batched matmul which is more efficient than stacking separate
  snt.Linear layers. This is implemented using `num_linear`s first to avoid the
  need for transposes in order to make it efficient when stacking these.
  """

  def __init__(self,
               output_size: int,
               with_bias: bool = True,
               w_init: Optional[initializers.Initializer] = None,
               b_init: Optional[initializers.Initializer] = None,
               name: Optional[str] = None):
    """Constructs a `ParallelLinear` module.

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
    super().__init__(name=name)
    self.output_size = output_size
    self.with_bias = with_bias
    self.w_init = w_init
    if with_bias:
      self.b_init = b_init if b_init is not None else initializers.Zeros()
    elif b_init is not None:
      raise ValueError("When not using a bias the b_init must be None.")

  @once.once
  def _initialize(self, inputs: tf.Tensor):
    """Constructs parameters used by this module."""
    utils.assert_rank(inputs, 3)

    self.input_size = inputs.shape[2]
    if self.input_size is None:  # Can happen inside an @tf.function.
      raise ValueError("Input size must be specified at module build time.")
    num_linears = inputs.shape[0]
    if num_linears is None:  # Can happen inside an @tf.function.
      raise ValueError(
          "The number of linears must be specified at module build time.")

    if self.w_init is None:
      # See https://arxiv.org/abs/1502.03167v3.
      stddev = 1. / math.sqrt(self.input_size)
      self.w_init = initializers.TruncatedNormal(stddev=stddev)

    self.w = tf.Variable(
        self.w_init([num_linears, self.input_size, self.output_size],
                    inputs.dtype),
        name="w")

    if self.with_bias:
      self.b = tf.Variable(
          self.b_init([num_linears, 1, self.output_size], inputs.dtype),
          name="b")

  def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
    self._initialize(inputs)

    outputs = tf.matmul(inputs, self.w)
    if self.with_bias:
      outputs = tf.add(outputs, self.b)
    return outputs
