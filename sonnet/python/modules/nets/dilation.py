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

"""Implementation of (Yu & Koltun, 2016)'s Dilation module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from sonnet.python.modules import base
from sonnet.python.modules import conv
from sonnet.python.modules import sequential
from sonnet.python.modules import util

import tensorflow as tf


def _range_along_dimension(range_dim, shape):
  """Construct a Tensor whose values are the index along a dimension.

  Construct a Tensor that counts the distance along a single dimension. This is
  useful, for example, when constructing an identity matrix,

    >>> x = _range_along_dimension(0, [2, 2]).eval()
    >>> x
    array([[0, 0],
           [1, 1]], dtype=int32)

    >>> y = _range_along_dimension(1, [2, 2]).eval()
    >>> y
    array([[0, 1],
           [0, 1]], dtype=int32)

    >>> tf.cast(tf.equal(x, y), dtype=tf.int32).eval()
    array([[1, 0],
           [0, 1]], dtype=int32)

  Args:
    range_dim: int. Dimension to count indices on.
    shape: 1D Tensor of ints. Shape of Tensor to construct.

  Returns:
    A Tensor whose values are the same as the range along dimension range_dim.

  Raises:
    ValueError: If range_dim isn't a valid dimension.
  """
  rank = len(shape)
  if range_dim >= rank:
    raise ValueError("Cannot calculate range along non-existent index.")
  indices = tf.range(start=0, limit=shape[range_dim])
  indices = tf.reshape(
      indices,
      shape=[1 if i != range_dim else shape[range_dim] for i in range(rank)])
  return tf.tile(indices,
                 [shape[i] if i != range_dim else 1 for i in range(rank)])


# pylint: disable=unused-argument
def identity_kernel_initializer(shape, dtype=tf.float32, partition_info=None):
  """An initializer for constructing identity convolution kernels.

  Constructs a convolution kernel such that applying it is the same as an
  identity operation on the input. Formally, the kernel has entry [i, j, in,
  out] = 1 if in equals out and i and j are the middle of the kernel and 0
  otherwise.

  Args:
    shape: List of integers. Represents shape of result.
    dtype: data type for values in result.
    partition_info: Partition information for initializer functions. Ignored.

  Returns:
    Tensor of desired shape and dtype such that applying it as a convolution
      kernel results in the identity operation.

  Raises:
    ValueError: If shape does not define a valid kernel.
                If filter width and height differ.
                If filter width and height are not odd numbers.
                If number of input and output channels differ.
  """
  if len(shape) != 4:
    raise ValueError("Convolution kernels must be rank 4.")

  filter_height, filter_width, in_channels, out_channels = shape

  if filter_width != filter_height:
    raise ValueError("Identity initializer only works for square filters.")
  if filter_width % 2 != 1:
    raise ValueError(
        "Identity initializer requires filters have odd height and width.")
  if in_channels != out_channels:
    raise ValueError(
        "in_channels must equal out_channels in order to construct per-channel"
        " identities.")

  middle_pixel = filter_height // 2
  is_middle_pixel = tf.logical_and(
      tf.equal(_range_along_dimension(0, shape), middle_pixel),
      tf.equal(_range_along_dimension(1, shape), middle_pixel))
  is_same_channel = tf.equal(
      _range_along_dimension(2, shape), _range_along_dimension(3, shape))
  return tf.cast(tf.logical_and(is_same_channel, is_middle_pixel), dtype=dtype)


def noisy_identity_kernel_initializer(base_num_channels, stddev=1e-8):
  """Build an initializer for constructing near-identity convolution kernels.

  Construct a convolution kernel where in_channels and out_channels are
  multiples of base_num_channels, but need not be equal. This initializer is
  essentially the same as identity_kernel_initializer, except that magnitude
  is "spread out" across multiple copies of the input.

  Args:
    base_num_channels: int. Number that divides both in_channels and
      out_channels.
    stddev: float. Standard deviation of truncated normal noise added to
      off-entries to break ties.

  Returns:
    Initializer function for building a noisy identity kernel.
  """

  # pylint: disable=unused-argument
  def _noisy_identity_kernel_initializer(shape,
                                         dtype=tf.float32,
                                         partition_info=None):
    """Constructs a noisy identity kernel.

    Args:
      shape: List of integers. Represents shape of result.
      dtype: data type for values in result.
      partition_info: Partition information for initializer functions. Ignored.

    Returns:
      Tensor of desired shape and dtype such that applying it as a convolution
        kernel results in a noisy near-identity operation.

    Raises:
      ValueError: If shape does not define a valid kernel.
                  If filter width and height differ.
                  If filter width and height are not odd numbers.
                  If number of input and output channels are not multiples of
                    base_num_channels.
    """
    if len(shape) != 4:
      raise ValueError("Convolution kernels must be rank 4.")

    filter_height, filter_width, in_channels, out_channels = shape

    if filter_width != filter_height:
      raise ValueError(
          "Noisy identity initializer only works for square filters.")
    if filter_width % 2 != 1:
      raise ValueError(
          "Noisy identity initializer requires filters have odd height and "
          "width.")
    if (in_channels % base_num_channels != 0 or
        out_channels % base_num_channels != 0):
      raise ValueError("in_channels and out_channels must both be multiples of "
                       "base_num_channels.")

    middle_pixel = filter_height // 2
    is_middle_pixel = tf.logical_and(
        tf.equal(_range_along_dimension(0, shape), middle_pixel),
        tf.equal(_range_along_dimension(1, shape), middle_pixel))
    is_same_channel_multiple = tf.equal(
        tf.floordiv(
            _range_along_dimension(2, shape) * base_num_channels, in_channels),
        tf.floordiv(
            _range_along_dimension(3, shape) * base_num_channels, out_channels))
    noise = tf.truncated_normal(shape, stddev=stddev, dtype=dtype)
    return tf.where(
        tf.logical_and(is_same_channel_multiple, is_middle_pixel),
        tf.ones(
            shape, dtype=dtype) * (base_num_channels / out_channels),
        noise)

  return _noisy_identity_kernel_initializer


class Dilation(base.AbstractModule):
  """A convolutional module for per-pixel classification.

  Consists of 8 convolutional layers, 4 of which are dilated. When applied to
  the output of a model like VGG-16 (before fully connected layers), can be used
  to make predictions on a per-pixel basis.

  Note that the default initializers for the 'basic' model size require that
  the number of input channels be equal to the number of output classes, and the
  initializers for the 'large' model require it be a multiple.

  Based on:
    'Multi-Scale Context Aggregation by Dilated Convolutions'
    Fisher Yu, Vladlen Koltun, ICLR 2016
    https://arxiv.org/abs/1511.07122

  Properties:
    conv_modules: list of sonnet modules. The 8 convolution layers used in the
      Dilation module.
  """

  # Size of model to build.
  BASIC = "basic"
  LARGE = "large"

  # Keys for initializers.
  WEIGHTS = "w"
  BIASES = "b"
  POSSIBLE_INITIALIZER_KEYS = {WEIGHTS, BIASES}

  def __init__(self,
               num_output_classes,
               initializers=None,
               regularizers=None,
               model_size="basic",
               name="dilation"):
    """Creates a dilation module.

    Args:
      num_output_classes: Int. Number of output classes to predict for
        each pixel in an image.
      initializers: Optional dict containing ops to initialize filters (with key
        'w') or biases (with key 'b'). The default initializer makes this module
        equivalent to the identity.
      regularizers: Optional dict containing regularizers for the weights
        (with key 'w') or biases (with key 'b'). As a default, no regularizers
        are used. A regularizer should be a function that takes a single
        `Tensor` as an input and returns a scalar `Tensor` output, e.g. the L1
        and L2 regularizers in `tf.contrib.layers`.
      model_size: string. One of 'basic' or 'large'.
      name: string. Name of module.
    """
    super(Dilation, self).__init__(name=name)
    self._num_output_classes = num_output_classes
    self._model_size = model_size
    self._initializers = util.check_initializers(
        initializers, self.POSSIBLE_INITIALIZER_KEYS)
    self._regularizers = util.check_regularizers(
        regularizers, self.POSSIBLE_INITIALIZER_KEYS)

  def _build(self, images):
    """Build dilation module.

    Args:
      images: Tensor of shape [batch_size, height, width, depth]
        and dtype float32. Represents a set of images with an arbitrary depth.
        Note that when using the default initializer, depth must equal
        num_output_classes.

    Returns:
      Tensor of shape [batch_size, height, width, num_output_classes] and dtype
        float32. Represents, for each image and pixel, logits for per-class
        predictions.

    Raises:
      IncompatibleShapeError: If images is not rank 4.
      ValueError: If model_size is not one of 'basic' or 'large'.
    """
    num_classes = self._num_output_classes

    if len(images.get_shape()) != 4:
      raise base.IncompatibleShapeError(
          "'images' must have shape [batch_size, height, width, depth].")

    if self.WEIGHTS not in self._initializers:
      if self._model_size == self.BASIC:
        self._initializers[self.WEIGHTS] = identity_kernel_initializer
      elif self._model_size == self.LARGE:
        self._initializers[self.WEIGHTS] = noisy_identity_kernel_initializer(
            num_classes)
      else:
        raise ValueError("Unrecognized model_size: %s" % self._model_size)

    if self.BIASES not in self._initializers:
      self._initializers[self.BIASES] = tf.zeros_initializer()

    if self._model_size == self.BASIC:
      self._conv_modules = [
          self._dilated_conv_layer(num_classes, 1, True, "conv1"),
          self._dilated_conv_layer(num_classes, 1, True, "conv2"),
          self._dilated_conv_layer(num_classes, 2, True, "conv3"),
          self._dilated_conv_layer(num_classes, 4, True, "conv4"),
          self._dilated_conv_layer(num_classes, 8, True, "conv5"),
          self._dilated_conv_layer(num_classes, 16, True, "conv6"),
          self._dilated_conv_layer(num_classes, 1, True, "conv7"),
          self._dilated_conv_layer(num_classes, 1, False, "conv8"),
      ]
    elif self._model_size == self.LARGE:
      self._conv_modules = [
          self._dilated_conv_layer(2 * num_classes, 1, True, "conv1"),
          self._dilated_conv_layer(2 * num_classes, 1, True, "conv2"),
          self._dilated_conv_layer(4 * num_classes, 2, True, "conv3"),
          self._dilated_conv_layer(8 * num_classes, 4, True, "conv4"),
          self._dilated_conv_layer(16 * num_classes, 8, True, "conv5"),
          self._dilated_conv_layer(32 * num_classes, 16, True, "conv6"),
          self._dilated_conv_layer(32 * num_classes, 1, True, "conv7"),
          self._dilated_conv_layer(num_classes, 1, False, "conv8"),
      ]
    else:
      raise ValueError("Unrecognized model_size: %s" % self._model_size)

    dilation_mod = sequential.Sequential(self._conv_modules, name="dilation")
    return dilation_mod(images)

  def _dilated_conv_layer(self, output_channels, dilation_rate, apply_relu,
                          name):
    """Create a dilated convolution layer.

    Args:
      output_channels: int. Number of output channels for each pixel.
      dilation_rate: int. Represents how many pixels each stride offset will
        move. A value of 1 indicates a standard convolution.
      apply_relu: bool. If True, a ReLU non-linearlity is added.
      name: string. Name for layer.

    Returns:
      a sonnet Module for a dilated convolution.
    """
    layer_components = [
        conv.Conv2D(
            output_channels, [3, 3],
            initializers=self._initializers,
            regularizers=self._regularizers,
            rate=dilation_rate,
            name="dilated_conv_" + name),
    ]
    if apply_relu:
      layer_components.append(lambda net: tf.nn.relu(net, name="relu_" + name))
    return sequential.Sequential(layer_components, name=name)

  @property
  def conv_modules(self):
    self._ensure_is_connected()
    return self._conv_modules
