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

"""Depth-wise convolutional module."""

from typing import Optional, Sequence, Union

import numpy as np
from sonnet.src import base
from sonnet.src import initializers
from sonnet.src import once
from sonnet.src import utils
import tensorflow as tf


class DepthwiseConv2D(base.Module):
  """Spatial depth-wise 2D convolution module, including bias.

  This acts as a light wrapper around the TensorFlow ops
  `tf.nn.depthwise_conv2d`, abstracting away variable creation and sharing.
  """

  def __init__(self,
               kernel_shape: Union[int, Sequence[int]],
               channel_multiplier: int = 1,
               stride: Union[int, Sequence[int]] = 1,
               rate: Union[int, Sequence[int]] = 1,
               padding: str = "SAME",
               with_bias: bool = True,
               w_init: Optional[initializers.Initializer] = None,
               b_init: Optional[initializers.Initializer] = None,
               data_format: str = "NHWC",
               name: Optional[str] = None):
    """Constructs a `DepthwiseConv2D` module.

    Args:
      kernel_shape: Sequence of kernel sizes (of length num_spatial_dims), or an
        integer. `kernel_shape` will be expanded to define a kernel size in
        all dimensions.
      channel_multiplier: Number of channels to expand convolution to. Must be
          an integer greater than 0. When `channel_multiplier` is 1, applies
          a different filter to each input channel producing one output channel
          per input channel. Numbers larger than 1 cause multiple different
          filters to be applied to each input channel, with their outputs being
          concatenated together, producing `channel_multiplier` *
          `input_channels` output channels.
      stride: Sequence of strides (of length num_spatial_dims), or an integer.
        `stride` will be expanded to define stride in all dimensions.
      rate: Sequence of dilation rates (of length num_spatial_dims), or integer
        that is used to define dilation rate in all dimensions. 1 corresponds
        to standard ND convolution, `rate > 1` corresponds to dilated
        convolution.
      padding: Padding to apply to the input. This can either "SAME", "VALID".
      with_bias: Whether to include bias parameters. Default `True`.
      w_init: Optional initializer for the weights. By default the weights are
        initialized truncated random normal values with a standard deviation of
        `1 / sqrt(input_feature_size)`, which is commonly used when the inputs
        are zero centered (see https://arxiv.org/abs/1502.03167v3).
      b_init: Optional initializer for the bias. By default the bias is
        initialized to zero.
      data_format: The data format of the input.
      name: Name of the module.
    """
    super().__init__(name=name)

    self.channel_multiplier = channel_multiplier
    self.kernel_shape = kernel_shape
    self.data_format = data_format
    self._channel_index = utils.get_channel_index(data_format)
    stride = utils.replicate(stride, 2, "stride")
    if self._channel_index == 1:
      self.stride = (1, 1) + stride
    else:
      self.stride = (1,) + stride + (1,)
    self.rate = utils.replicate(rate, 2, "rate")
    self.padding = padding

    self.with_bias = with_bias
    self.w_init = w_init
    if with_bias:
      self.b_init = b_init if b_init is not None else initializers.Zeros()
    elif b_init is not None:
      raise ValueError("When not using a bias the b_init must be None.")

  def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
    self._initialize(inputs)

    outputs = tf.nn.depthwise_conv2d(inputs,
                                     self.w,
                                     strides=self.stride,
                                     dilations=self.rate,
                                     padding=self.padding,
                                     data_format=self.data_format)
    if self.with_bias:
      outputs = tf.nn.bias_add(outputs, self.b,
                               data_format=self.data_format)

    return outputs

  @once.once
  def _initialize(self, inputs: tf.Tensor):
    self.input_channels = inputs.shape[self._channel_index]
    if self.input_channels is None:
      raise ValueError("The number of input channels must be known.")
    dtype = inputs.dtype

    weight_shape = utils.replicate(self.kernel_shape, 2, "kernel_shape")
    weight_shape = weight_shape + (self.input_channels, self.channel_multiplier)
    if self.w_init is None:
      # See https://arxiv.org/abs/1502.03167v3.
      fan_in_shape = weight_shape[:2]
      stddev = 1 / np.sqrt(np.prod(fan_in_shape))
      self.w_init = initializers.TruncatedNormal(stddev=stddev)
    self.w = tf.Variable(self.w_init(weight_shape, dtype), name="w")

    output_channels = self.input_channels * self.channel_multiplier
    if self.with_bias:
      self.b = tf.Variable(self.b_init((output_channels,), dtype), name="b")
