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
"""Convolutional modules."""

from typing import Optional, Sequence, Union

import numpy as np
from sonnet.src import base
from sonnet.src import initializers
from sonnet.src import once
from sonnet.src import pad
from sonnet.src import utils
import tensorflow as tf


class ConvND(base.Module):
  """A general N-dimensional convolutional module."""

  def __init__(self,
               num_spatial_dims: int,
               output_channels: int,
               kernel_shape: Union[int, Sequence[int]],
               stride: Union[int, Sequence[int]] = 1,
               rate: Union[int, Sequence[int]] = 1,
               padding: Union[str, pad.Paddings] = "SAME",
               with_bias: bool = True,
               w_init: Optional[initializers.Initializer] = None,
               b_init: Optional[initializers.Initializer] = None,
               data_format: Optional[str] = None,
               name: Optional[str] = None):
    """Constructs a `ConvND` module.

    Args:
      num_spatial_dims: The number of spatial dimensions of the input.
      output_channels: The number of output channels.
      kernel_shape: Sequence of kernel sizes (of length num_spatial_dims), or an
        integer. `kernel_shape` will be expanded to define a kernel size in all
        dimensions.
      stride: Sequence of strides (of length num_spatial_dims), or an integer.
        `stride` will be expanded to define stride in all dimensions.
      rate: Sequence of dilation rates (of length num_spatial_dims), or integer
        that is used to define dilation rate in all dimensions. 1 corresponds to
        standard ND convolution, `rate > 1` corresponds to dilated convolution.
      padding: Padding to apply to the input. This can either "SAME", "VALID" or
        a callable or sequence of callables up to size N. Any callables must
        take a single integer argument equal to the effective kernel size and
        return a list of two integers representing the padding before and after.
        See snt.pad.* for more details and example functions.
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

    if not 1 <= num_spatial_dims <= 3:
      raise ValueError(
          "We only support convoltion operations for num_spatial_dims=1, 2 or "
          "3, received num_spatial_dims={}.".format(num_spatial_dims))
    self._num_spatial_dims = num_spatial_dims
    self.output_channels = output_channels
    self.kernel_shape = kernel_shape
    self.stride = stride
    self.rate = rate

    if isinstance(padding, str):
      self.conv_padding = padding.upper()
      self.padding_func = None
    else:
      self.conv_padding = "VALID"
      self.padding_func = padding

    self.data_format = data_format
    self._channel_index = utils.get_channel_index(data_format)
    self.with_bias = with_bias

    self.w_init = w_init
    if with_bias:
      self.b_init = b_init if b_init is not None else initializers.Zeros()
    elif b_init is not None:
      raise ValueError("When not using a bias the b_init must be None.")

  def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
    """Applies the defined convolution to the inputs.

    Args:
      inputs: An ``N + 2`` rank :tf:`Tensor` of dtype :tf:`float16`,
        :tf:`bfloat16` or `tf.float32` to which the convolution is applied.

    Returns:
      An ``N + 2`` dimensional :tf:`Tensor` of shape
        ``[batch_size, output_dim_1, output_dim_2, ..., output_channels]``.
    """
    self._initialize(inputs)

    if self.padding_func:
      inputs = tf.pad(inputs, self._padding)

    outputs = tf.nn.convolution(
        inputs,
        self.w,
        strides=self.stride,
        padding=self.conv_padding,
        dilations=self.rate,
        data_format=self.data_format)
    if self.with_bias:
      outputs = tf.nn.bias_add(outputs, self.b, data_format=self.data_format)

    return outputs

  @once.once
  def _initialize(self, inputs: tf.Tensor):
    """Constructs parameters used by this module."""
    utils.assert_rank(inputs, self._num_spatial_dims + 2)
    self.input_channels = inputs.shape[self._channel_index]
    if self.input_channels is None:
      raise ValueError("The number of input channels must be known.")
    self._dtype = inputs.dtype

    self.w = self._make_w()
    if self.with_bias:
      self.b = tf.Variable(
          self.b_init((self.output_channels,), self._dtype), name="b")

    if self.padding_func:
      self._padding = pad.create(
          padding=self.padding_func,
          kernel=self.kernel_shape,
          rate=self.rate,
          n=self._num_spatial_dims,
          channel_index=self._channel_index)

  def _make_w(self):
    weight_shape = utils.replicate(self.kernel_shape, self._num_spatial_dims,
                                   "kernel_shape")
    weight_shape = weight_shape + (self.input_channels, self.output_channels)

    if self.w_init is None:
      # See https://arxiv.org/abs/1502.03167v3.
      fan_in_shape = weight_shape[:-1]
      stddev = 1 / np.sqrt(np.prod(fan_in_shape))
      self.w_init = initializers.TruncatedNormal(stddev=stddev)

    return tf.Variable(self.w_init(weight_shape, self._dtype), name="w")


class Conv1D(ConvND):
  """``Conv1D`` module."""

  def __init__(self,
               output_channels: int,
               kernel_shape: Union[int, Sequence[int]],
               stride: Union[int, Sequence[int]] = 1,
               rate: Union[int, Sequence[int]] = 1,
               padding: Union[str, pad.Paddings] = "SAME",
               with_bias: bool = True,
               w_init: Optional[initializers.Initializer] = None,
               b_init: Optional[initializers.Initializer] = None,
               data_format: str = "NWC",
               name: Optional[str] = None):
    """Constructs a ``Conv1D`` module.

    Args:
      output_channels: The number of output channels.
      kernel_shape: Sequence of length 1, or an integer. ``kernel_shape`` will
        be expanded to define a kernel size in all dimensions.
      stride: Sequence of strides of length 1, or an integer. ``stride`` will be
        expanded to define stride in all dimensions.
      rate: Sequence of dilation rates of length 1, or integer that is used to
        define dilation rate in all dimensions. 1 corresponds to standard
        convolution, ``rate > 1`` corresponds to dilated convolution.
      padding: Padding to apply to the input. This can be either ``SAME``,
        ``VALID`` or a callable or sequence of callables of size 1. Any
        callables must take a single integer argument equal to the effective
        kernel size and return a list of two integers representing the padding
        before and after. See snt.pad.* for more details and example functions.
      with_bias: Whether to include bias parameters. Default ``True``.
      w_init: Optional initializer for the weights. By default the weights are
        initialized truncated random normal values with a standard deviation of
        ``1``/``sqrt(input_feature_size)``, which is commonly used when the
        inputs are zero centered (see https://arxiv.org/abs/1502.03167v3).
      b_init: Optional initializer for the bias. By default the bias is
        initialized to zero.
      data_format: The data format of the input.
      name: Name of the module.
    """
    super().__init__(
        num_spatial_dims=1,
        output_channels=output_channels,
        kernel_shape=kernel_shape,
        stride=stride,
        rate=rate,
        padding=padding,
        with_bias=with_bias,
        w_init=w_init,
        b_init=b_init,
        data_format=data_format,
        name=name)


class Conv2D(ConvND):
  """`Conv2D` module."""

  def __init__(self,
               output_channels: int,
               kernel_shape: Union[int, Sequence[int]],
               stride: Union[int, Sequence[int]] = 1,
               rate: Union[int, Sequence[int]] = 1,
               padding: Union[str, pad.Paddings] = "SAME",
               with_bias: bool = True,
               w_init: Optional[initializers.Initializer] = None,
               b_init: Optional[initializers.Initializer] = None,
               data_format: str = "NHWC",
               name: Optional[str] = None):
    """Constructs a ``Conv2D`` module.

    Args:
      output_channels: The number of output channels.
      kernel_shape: Sequence of kernel sizes (of length 2), or an integer.
        ``kernel_shape`` will be expanded to define a kernel size in all
        dimensions.
      stride: Sequence of strides (of length 2), or an integer. ``stride`` will
        be expanded to define stride in all dimensions.
      rate: Sequence of dilation rates (of length 2), or integer that is used to
        define dilation rate in all dimensions. 1 corresponds to standard
        convolution, ``rate > 1`` corresponds to dilated convolution.
      padding: Padding to apply to the input. This can either ``SAME``,
        ``VALID`` or a callable or sequence of callables of size 2. Any
        callables must take a single integer argument equal to the effective
        kernel size and return a list of two integers representing the padding
        before and after. See snt.pad.* for more details and example functions.
      with_bias: Whether to include bias parameters. Default ``True``.
      w_init: Optional initializer for the weights. By default the weights are
        initialized truncated random normal values with a standard deviation of
        ``1 / sqrt(input_feature_size)``, which is commonly used when the inputs
        are zero centered (see https://arxiv.org/abs/1502.03167v3).
      b_init: Optional initializer for the bias. By default the bias is
        initialized to zero.
      data_format: The data format of the input.
      name: Name of the module.
    """
    super().__init__(
        num_spatial_dims=2,
        output_channels=output_channels,
        kernel_shape=kernel_shape,
        stride=stride,
        rate=rate,
        padding=padding,
        with_bias=with_bias,
        w_init=w_init,
        b_init=b_init,
        data_format=data_format,
        name=name)


class Conv3D(ConvND):
  """`Conv3D` module."""

  def __init__(self,
               output_channels: int,
               kernel_shape: Union[int, Sequence[int]],
               stride: Union[int, Sequence[int]] = 1,
               rate: Union[int, Sequence[int]] = 1,
               padding: Union[str, pad.Paddings] = "SAME",
               with_bias: bool = True,
               w_init: Optional[initializers.Initializer] = None,
               b_init: Optional[initializers.Initializer] = None,
               data_format: str = "NDHWC",
               name: Optional[str] = None):
    """Constructs a ``Conv3D`` module.

    Args:
      output_channels: The number of output channels.
      kernel_shape: Sequence of kernel sizes (of length 3), or an integer.
        ``kernel_shape`` will be expanded to define a kernel size in all
        dimensions.
      stride: Sequence of strides (of length 3), or an integer. `stride` will be
        expanded to define stride in all dimensions.
      rate: Sequence of dilation rates (of length 3), or integer that is used to
        define dilation rate in all dimensions. 1 corresponds to standard
        convolution, ``rate > 1`` corresponds to dilated convolution.
      padding: Padding to apply to the input. This can either ``SAME``,
        ``VALID`` or a callable or sequence of callables up to size N. Any
        callables must take a single integer argument equal to the effective
        kernel size and return a list of two integers representing the padding
        before and after. See snt.pad.* for more details and example functions.
      with_bias: Whether to include bias parameters. Default ``True``.
      w_init: Optional initializer for the weights. By default the weights are
        initialized truncated random normal values with a standard deviation of
        ``1 / sqrt(input_feature_size)``, which is commonly used when the inputs
        are zero centered (see https://arxiv.org/abs/1502.03167v3).
      b_init: Optional initializer for the bias. By default the bias is
        initialized to zero.
      data_format: The data format of the input.
      name: Name of the module.
    """
    super().__init__(
        num_spatial_dims=3,
        output_channels=output_channels,
        kernel_shape=kernel_shape,
        stride=stride,
        rate=rate,
        padding=padding,
        with_bias=with_bias,
        w_init=w_init,
        b_init=b_init,
        data_format=data_format,
        name=name)
