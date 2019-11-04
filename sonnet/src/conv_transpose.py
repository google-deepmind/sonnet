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
"""Transpose convolutional module."""

from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

import numpy as np

from sonnet.src import base
from sonnet.src import initializers
from sonnet.src import once
from sonnet.src import types
from sonnet.src import utils

import tensorflow as tf
from typing import Optional, Sequence, Text, Union


class ConvNDTranspose(base.Module):
  """An N-dimensional transpose convolutional module.

  Attributes:
     w: Weight variable. Note is `None` until module is connected.
     b: Biases variable. Note is `None` until module is connected.
     input_shape: The input shape of the first set of inputs. Note is `None`
       until module is connected.
  """

  def __init__(self,
               num_spatial_dims: int,
               output_channels: int,
               kernel_shape: Union[int, Sequence[int]],
               output_shape: Optional[types.ShapeLike] = None,
               stride: Union[int, Sequence[int]] = 1,
               rate: Union[int, Sequence[int]] = 1,
               padding: Text = "SAME",
               with_bias: bool = True,
               w_init: Optional[initializers.Initializer] = None,
               b_init: Optional[initializers.Initializer] = None,
               data_format: Optional[Text] = None,
               name: Optional[Text] = None):
    """Constructs a `ConvNDTranspose` module.

    Args:
      num_spatial_dims: Number of spatial dimensions of the input.
      output_channels: Number of output channels.
      kernel_shape: Sequence of integers (of length num_spatial_dims), or an
        integer representing kernel shape. `kernel_shape` will be expanded to
        define a kernel size in all dimensions.
      output_shape: Output shape of the spatial dimensions of a transpose
        convolution. Can be either an integer or an iterable of integers or a
        `TensorShape` of length `num_spatial_dims`. If a `None` value is given,
        a default shape is automatically calculated.
      stride: Sequence of integers (of length num_spatial_dims), or an integer.
        `stride` will be expanded to define stride in all dimensions.
      rate: Sequence of integers (of length num_spatial_dims), or integer that
        is used to define dilation rate in all dimensions. 1 corresponds to
        standard ND convolution, `rate > 1` corresponds to dilated convolution.
      padding: Padding algorithm, either "SAME" or "VALID".
      with_bias: Boolean, whether to include bias parameters. Default `True`.
      w_init: Optional initializer for the weights. By default the weights are
        initialized truncated random normal values with a standard deviation of
        `1 / sqrt(input_feature_size)`, which is commonly used when the
        inputs are zero centered (see https://arxiv.org/abs/1502.03167v3).
      b_init: Optional initializer for the bias. By default the bias is
        initialized to zero.
      data_format: The data format of the input.
      name: Name of the module.
    """
    super(ConvNDTranspose, self).__init__(name=name)

    if not 1 <= num_spatial_dims <= 3:
      raise ValueError(
          "We only support transpose convolution operations for "
          "num_spatial_dims=1, 2 or 3, received num_spatial_dims={}.".format(
              num_spatial_dims))
    self._num_spatial_dims = num_spatial_dims
    self._output_channels = output_channels
    self._kernel_shape = kernel_shape
    self._output_shape = output_shape
    self._stride = stride
    self._rate = rate

    if padding == "SAME" or padding == "VALID":
      self._padding = padding
    else:
      raise TypeError("ConvNDTranspose only takes string padding, please "
                      "provide either `SAME` or `VALID`.")
    self._data_format = data_format
    self._channel_index = utils.get_channel_index(data_format)
    self._with_bias = with_bias

    self._w_init = w_init
    if with_bias:
      self._b_init = b_init if b_init is not None else initializers.Zeros()
    elif b_init is not None:
      raise ValueError("When not using a bias the b_init must be None.")

  def __call__(self, inputs):
    self._initialize(inputs)

    output_shape = tf.concat([[tf.shape(inputs)[0]], self._output_shape], 0)

    outputs = tf.nn.conv_transpose(
        input=inputs,
        filters=self.w,
        output_shape=output_shape,
        strides=self._stride,
        padding=self._padding,
        data_format=self._data_format,
        dilations=self._rate,
        name=None)
    if self._with_bias:
      outputs = tf.nn.bias_add(outputs, self.b, data_format=self._data_format)
    return outputs

  @once.once
  def _initialize(self, inputs):
    utils.assert_rank(inputs, self._num_spatial_dims + 2)
    self.input_channels = inputs.shape[self._channel_index]
    if self.input_channels is None:
      raise ValueError("The number of input channels must be known")
    self._dtype = inputs.dtype

    if self._output_shape is None:
      self._output_shape = self._get_output_shape(inputs.shape)
    elif len(self._output_shape) != self._num_spatial_dims:
      raise ValueError(
          "The output_shape must be of length {} but instead was {}".format(
              self._n, len(self._output_shape)))

    if self._channel_index == 1:
      self._output_shape = (self._output_channels,) + self._output_shape
    else:
      self._output_shape = self._output_shape + (self._output_channels,)

    self.w = self._make_w()
    if self._with_bias:
      self.b = tf.Variable(
          self._b_init((self._output_channels,), self._dtype), name="b")

  def _make_w(self):
    """Makes and returns the variable representing the weight."""
    kernel_shape = utils.replicate(self._kernel_shape, self._num_spatial_dims,
                                   "kernel_shape")
    weight_shape = kernel_shape + (self._output_channels, self.input_channels)

    if self._w_init is None:
      # See https://arxiv.org/abs/1502.03167v3.
      fan_in_shape = kernel_shape + (self.input_channels,)
      stddev = 1 / np.sqrt(np.prod(fan_in_shape))
      self._w_init = initializers.TruncatedNormal(stddev=stddev)

    return tf.Variable(self._w_init(weight_shape, self._dtype), name="w")

  def _get_output_shape(self, input_shape):
    if self._channel_index == 1:
      input_size = input_shape[2:]
    else:
      input_size = input_shape[1:-1]
    stride = utils.replicate(self._stride, self._num_spatial_dims, "stride")

    output_shape = [x * y for (x, y) in zip(input_size, stride)]

    if self._padding == "VALID":
      kernel_shape = utils.replicate(self._kernel_shape, self._num_spatial_dims,
                                     "kernel_shape")
      rate = utils.replicate(self._rate, self._num_spatial_dims, "rate")
      effective_kernel_shape = [
          (shape - 1) * rate + 1 for (shape, rate) in zip(kernel_shape, rate)
      ]
      output_shape = [
          x + y - 1 for (x, y) in zip(output_shape, effective_kernel_shape)
      ]

    return tuple(output_shape)


class Conv1DTranspose(ConvNDTranspose):
  """A 1D transpose convolutional module."""

  def __init__(self,
               output_channels: int,
               kernel_shape: Union[int, Sequence[int]],
               output_shape: Optional[types.ShapeLike] = None,
               stride: Union[int, Sequence[int]] = 1,
               rate: Union[int, Sequence[int]] = 1,
               padding: Text = "SAME",
               with_bias: bool = True,
               w_init: Optional[initializers.Initializer] = None,
               b_init: Optional[initializers.Initializer] = None,
               data_format: Text = "NWC",
               name: Optional[Text] = None):
    """Constructs a `Conv1DTranspose` module.

    Args:
      output_channels: Number of output channels.
      kernel_shape: Sequence of integers (of length 1), or an integer
        representing kernel shape. `kernel_shape` will be expanded to define a
        kernel size in all dimensions.
      output_shape: Output shape of the spatial dimensions of a transpose
        convolution. Can be either an integer or an iterable of integers or
        `Dimension`s, or a `TensorShape` (of length 1). If a `None` value is
        given, a default shape is automatically calculated.
      stride: Sequence of integers (of length 1), or an integer. `stride` will
        be expanded to define stride in all dimensions.
      rate: Sequence of integers (of length 1), or integer that is used to
        define dilation rate in all dimensions. 1 corresponds to standard 1D
        convolution, `rate > 1` corresponds to dilated convolution.
      padding: Padding algorithm, either "SAME" or "VALID".
      with_bias: Boolean, whether to include bias parameters. Default `True`.
      w_init: Optional initializer for the weights. By default the weights are
        initialized truncated random normal values with a standard deviation of
        `1 / sqrt(input_feature_size)`, which is commonly used when the
        inputs are zero centered (see https://arxiv.org/abs/1502.03167v3).
      b_init: Optional initializer for the bias. By default the bias is
        initialized to zero.
      data_format: The data format of the input.
      name: Name of the module.
    """
    super(Conv1DTranspose, self).__init__(
        num_spatial_dims=1,
        output_channels=output_channels,
        kernel_shape=kernel_shape,
        output_shape=output_shape,
        stride=stride,
        rate=rate,
        padding=padding,
        with_bias=with_bias,
        w_init=w_init,
        b_init=b_init,
        data_format=data_format,
        name=name)


class Conv2DTranspose(ConvNDTranspose):
  """A 2D transpose convolutional module."""

  def __init__(self,
               output_channels: int,
               kernel_shape: Union[int, Sequence[int]],
               output_shape: Optional[types.ShapeLike] = None,
               stride: Union[int, Sequence[int]] = 1,
               rate: Union[int, Sequence[int]] = 1,
               padding: Text = "SAME",
               with_bias: bool = True,
               w_init: Optional[initializers.Initializer] = None,
               b_init: Optional[initializers.Initializer] = None,
               data_format: Text = "NHWC",
               name: Optional[Text] = None):
    """Constructs a `Conv2DTranspose` module.

    Args:
      output_channels: An integer, The number of output channels.
      kernel_shape: Sequence of integers (of length 2), or an integer
        representing kernel shape. `kernel_shape` will be expanded to define a
        kernel size in all dimensions.
      output_shape: Output shape of the spatial dimensions of a transpose
        convolution. Can be either an integer or an iterable of integers or
        `Dimension`s, or a `TensorShape` (of length 2). If a `None` value is
        given, a default shape is automatically calculated.
      stride: Sequence of integers (of length 2), or an integer. `stride` will
        be expanded to define stride in all dimensions.
      rate: Sequence of integers (of length 2), or integer that is used to
        define dilation rate in all dimensions. 1 corresponds to standard 2D
        convolution, `rate > 1` corresponds to dilated convolution.
      padding: Padding algorithm, either "SAME" or "VALID".
      with_bias: Boolean, whether to include bias parameters. Default `True`.
      w_init: Optional initializer for the weights. By default the weights are
        initialized truncated random normal values with a standard deviation of
        `1 / sqrt(input_feature_size)`, which is commonly used when the
        inputs are zero centered (see https://arxiv.org/abs/1502.03167v3).
      b_init: Optional initializer for the bias. By default the bias is
        initialized to zero.
      data_format: The data format of the input.
      name: Name of the module.
    """
    super(Conv2DTranspose, self).__init__(
        num_spatial_dims=2,
        output_channels=output_channels,
        kernel_shape=kernel_shape,
        output_shape=output_shape,
        stride=stride,
        rate=rate,
        padding=padding,
        with_bias=with_bias,
        w_init=w_init,
        b_init=b_init,
        data_format=data_format,
        name=name)


class Conv3DTranspose(ConvNDTranspose):
  """A 3D transpose convolutional module."""

  def __init__(self,
               output_channels: int,
               kernel_shape: Union[int, Sequence[int]],
               output_shape: Optional[types.ShapeLike] = None,
               stride: Union[int, Sequence[int]] = 1,
               rate: Union[int, Sequence[int]] = 1,
               padding: Text = "SAME",
               with_bias: bool = True,
               w_init: Optional[initializers.Initializer] = None,
               b_init: Optional[initializers.Initializer] = None,
               data_format: Text = "NDHWC",
               name: Optional[Text] = None):
    """Constructs a `Conv3DTranspose` module.

    Args:
      output_channels: An integer, The number of output channels.
      kernel_shape: Sequence of integers (of length 3), or an integer
        representing kernel shape. `kernel_shape` will be expanded to define a
        kernel size in all dimensions.
      output_shape: Output shape of the spatial dimensions of a transpose
        convolution. Can be either an integer or an iterable of integers or
        `Dimension`s, or a `TensorShape` (of length 3). If a None value is
        given, a default shape is automatically calculated.
      stride: Sequence of integers (of length 3), or an integer. `stride` will
        be expanded to define stride in all dimensions.
      rate: Sequence of integers (of length 3), or integer that is used to
        define dilation rate in all dimensions. 1 corresponds to standard 3D
        convolution, `rate > 1` corresponds to dilated convolution.
      padding: Padding algorithm, either "SAME" or "VALID".
      with_bias: Boolean, whether to include bias parameters. Default `True`.
      w_init: Optional initializer for the weights. By default the weights are
        initialized truncated random normal values with a standard deviation of
        `1 / sqrt(input_feature_size)`, which is commonly used when the
        inputs are zero centered (see https://arxiv.org/abs/1502.03167v3).
      b_init: Optional initializer for the bias. By default the bias is
        initialized to zero.
      data_format: The data format of the input.
      name: Name of the module.
    """
    super(Conv3DTranspose, self).__init__(
        num_spatial_dims=3,
        output_channels=output_channels,
        kernel_shape=kernel_shape,
        output_shape=output_shape,
        stride=stride,
        rate=rate,
        padding=padding,
        with_bias=with_bias,
        w_init=w_init,
        b_init=b_init,
        data_format=data_format,
        name=name)
