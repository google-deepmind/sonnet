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
"""Convnet module for Cifar10 classification."""

from typing import Mapping, Optional, Sequence, Union

from sonnet.src import base
from sonnet.src import batch_norm
from sonnet.src import conv
from sonnet.src import initializers
from sonnet.src import linear
from sonnet.src import types
import tensorflow as tf


class Cifar10ConvNet(base.Module):
  """Convolutional network designed for Cifar10.

  Approximately equivalent to "VGG, minus max pooling, plus BatchNorm". For best
  results the input data should be scaled to be between -1 and 1 when using the
  standard initializers.
  """

  def __init__(self,
               num_classes: int = 10,
               w_init: Optional[initializers.Initializer] = None,
               b_init: Optional[initializers.Initializer] = None,
               data_format: str = 'NHWC',
               output_channels: Sequence[int] = (
                   64,
                   64,
                   128,
                   128,
                   128,
                   256,
                   256,
                   256,
                   512,
                   512,
                   512,
               ),
               strides: Sequence[int] = (1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1),
               name: Optional[str] = None):
    super().__init__(name=name)
    self._num_classes = num_classes
    self._data_format = data_format
    if len(strides) != len(output_channels):
      raise ValueError(
          'The length of `output_channels` and `strides` must be equal.')
    self._output_channels = output_channels
    self._strides = strides
    self._num_layers = len(self._output_channels)
    self._kernel_shapes = [[3, 3]] * self._num_layers  # All kernels are 3x3.
    self._w_init = w_init
    self._b_init = b_init

    self._conv_modules = list(
        conv.Conv2D(  # pylint: disable=g-complex-comprehension
            output_channels=self._output_channels[i],
            kernel_shape=self._kernel_shapes[i],
            stride=self._strides[i],
            w_init=self._w_init,
            b_init=self._b_init,
            data_format=self._data_format,
            name='conv_2d_{}'.format(i)) for i in range(self._num_layers))
    self._bn_modules = list(
        batch_norm.BatchNorm(  # pylint: disable=g-complex-comprehension
            create_offset=True,
            create_scale=False,
            decay_rate=0.999,
            data_format=self._data_format,
            name='batch_norm_{}'.format(i)) for i in range(self._num_layers))
    self._logits_module = linear.Linear(
        self._num_classes,
        w_init=self._w_init,
        b_init=self._b_init,
        name='logits')

  def __call__(
      self,
      inputs: tf.Tensor,
      is_training: types.BoolLike,
      test_local_stats: bool = True
  ) -> Mapping[str, Union[tf.Tensor, Sequence[tf.Tensor]]]:
    """Connects the module to some inputs.

    Args:
      inputs: A Tensor of size [batch_size, input_height, input_width,
        input_channels], representing a batch of input images.
      is_training: Boolean to indicate to `snt.BatchNorm` if we are currently
        training.
      test_local_stats: Boolean to indicate to `snt.BatchNorm` if batch
        normalization should  use local batch statistics at test time. By
        default `True`.

    Returns:
      A dictionary containing two items:
      - logits: The output logits of the network, this will be of size
        [batch_size, num_classes]
      - activations: A list of `tf.Tensor`, the feature activations of the
        module. The order of the activations is preserved in the output list.
        The activations in the output list are those computed after the
        activation function is applied, if one is applied at that layer.
    """
    activations = []
    net = inputs
    for conv_layer, bn_layer in zip(self._conv_modules, self._bn_modules):
      net = conv_layer(net)
      net = bn_layer(
          net, is_training=is_training, test_local_stats=test_local_stats)
      net = tf.nn.relu(net)
      activations.append(net)

    flat_output = tf.reduce_mean(
        net, axis=[1, 2], keepdims=False, name='avg_pool')
    activations.append(flat_output)

    logits = self._logits_module(flat_output)

    return {'logits': logits, 'activations': activations}
