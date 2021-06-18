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
"""ResNet model for Sonnet."""

from typing import Mapping, Optional, Sequence, Union

from sonnet.src import base
from sonnet.src import batch_norm
from sonnet.src import conv
from sonnet.src import initializers
from sonnet.src import linear
from sonnet.src import pad
import tensorflow as tf


class BottleNeckBlockV1(base.Module):
  """Bottleneck Block for a ResNet implementation."""

  def __init__(self,
               channels: int,
               stride: Union[int, Sequence[int]],
               use_projection: bool,
               bn_config: Mapping[str, float],
               name: Optional[str] = None):
    super().__init__(name=name)
    self._channels = channels
    self._stride = stride
    self._use_projection = use_projection
    self._bn_config = bn_config

    batchnorm_args = {"create_scale": True, "create_offset": True}
    batchnorm_args.update(bn_config)

    if self._use_projection:
      self._proj_conv = conv.Conv2D(
          output_channels=channels,
          kernel_shape=1,
          stride=stride,
          with_bias=False,
          padding=pad.same,
          name="shortcut_conv")
      self._proj_batchnorm = batch_norm.BatchNorm(
          name="shortcut_batchnorm", **batchnorm_args)

    self._layers = []
    conv_0 = conv.Conv2D(
        output_channels=channels // 4,
        kernel_shape=1,
        stride=1,
        with_bias=False,
        padding=pad.same,
        name="conv_0")
    self._layers.append(
        [conv_0,
         batch_norm.BatchNorm(name="batchnorm_0", **batchnorm_args)])

    conv_1 = conv.Conv2D(
        output_channels=channels // 4,
        kernel_shape=3,
        stride=stride,
        with_bias=False,
        padding=pad.same,
        name="conv_1")
    self._layers.append(
        [conv_1,
         batch_norm.BatchNorm(name="batchnorm_1", **batchnorm_args)])

    conv_2 = conv.Conv2D(
        output_channels=channels,
        kernel_shape=1,
        stride=1,
        with_bias=False,
        padding=pad.same,
        name="conv_2")
    batchnorm_2 = batch_norm.BatchNorm(
        name="batchnorm_2", scale_init=initializers.Zeros(), **batchnorm_args)
    self._layers.append([conv_2, batchnorm_2])

  def __call__(self, inputs, is_training):
    if self._use_projection:
      shortcut = self._proj_conv(inputs)
      shortcut = self._proj_batchnorm(shortcut, is_training=is_training)
    else:
      shortcut = inputs

    net = inputs
    for i, [conv_layer, batchnorm_layer] in enumerate(self._layers):
      net = conv_layer(net)
      net = batchnorm_layer(net, is_training=is_training)
      net = tf.nn.relu(net) if i < 2 else net  # Don't apply relu on last layer

    return tf.nn.relu(net + shortcut)


class BottleNeckBlockV2(base.Module):
  """Bottleneck Block for a Resnet implementation."""

  def __init__(self,
               channels: int,
               stride: Union[int, Sequence[int]],
               use_projection: bool,
               bn_config: Mapping[str, float],
               name: Optional[str] = None):
    super().__init__(name=name)
    self._channels = channels
    self._stride = stride
    self._use_projection = use_projection
    self._bn_config = bn_config

    batchnorm_args = {"create_scale": True, "create_offset": True}
    batchnorm_args.update(bn_config)

    if self._use_projection:
      self._proj_conv = conv.Conv2D(
          output_channels=channels,
          kernel_shape=1,
          stride=stride,
          with_bias=False,
          padding=pad.same,
          name="shortcut_conv")

    self._conv_0 = conv.Conv2D(
        output_channels=channels // 4,
        kernel_shape=1,
        stride=1,
        with_bias=False,
        padding=pad.same,
        name="conv_0")

    self._bn_0 = batch_norm.BatchNorm(name="batchnorm_0", **batchnorm_args)

    self._conv_1 = conv.Conv2D(
        output_channels=channels // 4,
        kernel_shape=3,
        stride=stride,
        with_bias=False,
        padding=pad.same,
        name="conv_1")

    self._bn_1 = batch_norm.BatchNorm(name="batchnorm_1", **batchnorm_args)

    self._conv_2 = conv.Conv2D(
        output_channels=channels,
        kernel_shape=1,
        stride=1,
        with_bias=False,
        padding=pad.same,
        name="conv_2")

    # NOTE: Some implementations of ResNet50 v2 suggest initializing gamma/scale
    # here to zeros.
    self._bn_2 = batch_norm.BatchNorm(name="batchnorm_2", **batchnorm_args)

  def __call__(self, inputs, is_training):
    net = inputs
    shortcut = inputs

    for i, (conv_i, bn_i) in enumerate(((self._conv_0, self._bn_0),
                                        (self._conv_1, self._bn_1),
                                        (self._conv_2, self._bn_2))):
      net = bn_i(net, is_training=is_training)
      net = tf.nn.relu(net)
      if i == 0 and self._use_projection:
        shortcut = self._proj_conv(net)
      net = conv_i(net)

    return net + shortcut


class BlockGroup(base.Module):
  """Higher level block for ResNet implementation."""

  def __init__(self,
               channels: int,
               num_blocks: int,
               stride: Union[int, Sequence[int]],
               bn_config: Mapping[str, float],
               resnet_v2: bool = False,
               name: Optional[str] = None):
    super().__init__(name=name)
    self._channels = channels
    self._num_blocks = num_blocks
    self._stride = stride
    self._bn_config = bn_config

    if resnet_v2:
      bottle_neck_block = BottleNeckBlockV2
    else:
      bottle_neck_block = BottleNeckBlockV1

    self._blocks = []
    for id_block in range(num_blocks):
      self._blocks.append(
          bottle_neck_block(
              channels=channels,
              stride=stride if id_block == 0 else 1,
              use_projection=(id_block == 0),
              bn_config=bn_config,
              name="block_%d" % (id_block)))

  def __call__(self, inputs, is_training):
    net = inputs
    for block in self._blocks:
      net = block(net, is_training=is_training)
    return net


class ResNet(base.Module):
  """ResNet model."""

  def __init__(self,
               blocks_per_group_list: Sequence[int],
               num_classes: int,
               bn_config: Optional[Mapping[str, float]] = None,
               resnet_v2: bool = False,
               channels_per_group_list: Sequence[int] = (256, 512, 1024, 2048),
               name: Optional[str] = None):
    """Constructs a ResNet model.

    Args:
      blocks_per_group_list: A sequence of length 4 that indicates the number of
        blocks created in each group.
      num_classes: The number of classes to classify the inputs into.
      bn_config: A dictionary of two elements, `decay_rate` and `eps` to be
        passed on to the `BatchNorm` layers. By default the `decay_rate` is
        `0.9` and `eps` is `1e-5`.
      resnet_v2: Whether to use the v1 or v2 ResNet implementation. Defaults to
        False.
      channels_per_group_list: A sequence of length 4 that indicates the number
        of channels used for each block in each group.
      name: Name of the module.
    """
    super().__init__(name=name)
    if bn_config is None:
      bn_config = {"decay_rate": 0.9, "eps": 1e-5}
    self._bn_config = bn_config
    self._resnet_v2 = resnet_v2

    # Number of blocks in each group for ResNet.
    if len(blocks_per_group_list) != 4:
      raise ValueError(
          "`blocks_per_group_list` must be of length 4 not {}".format(
              len(blocks_per_group_list)))
    self._blocks_per_group_list = blocks_per_group_list

    # Number of channels in each group for ResNet.
    if len(channels_per_group_list) != 4:
      raise ValueError(
          "`channels_per_group_list` must be of length 4 not {}".format(
              len(channels_per_group_list)))
    self._channels_per_group_list = channels_per_group_list

    self._initial_conv = conv.Conv2D(
        output_channels=64,
        kernel_shape=7,
        stride=2,
        with_bias=False,
        padding=pad.same,
        name="initial_conv")
    if not self._resnet_v2:
      self._initial_batchnorm = batch_norm.BatchNorm(
          create_scale=True,
          create_offset=True,
          name="initial_batchnorm",
          **bn_config)

    self._block_groups = []
    strides = [1, 2, 2, 2]
    for i in range(4):
      self._block_groups.append(
          BlockGroup(
              channels=self._channels_per_group_list[i],
              num_blocks=self._blocks_per_group_list[i],
              stride=strides[i],
              bn_config=bn_config,
              resnet_v2=resnet_v2,
              name="block_group_%d" % (i)))

    if self._resnet_v2:
      self._final_batchnorm = batch_norm.BatchNorm(
          create_scale=True,
          create_offset=True,
          name="final_batchnorm",
          **bn_config)

    self._logits = linear.Linear(
        output_size=num_classes, w_init=initializers.Zeros(), name="logits")

  def __call__(self, inputs, is_training):
    net = inputs
    net = self._initial_conv(net)
    if not self._resnet_v2:
      net = self._initial_batchnorm(net, is_training=is_training)
      net = tf.nn.relu(net)

    net = tf.nn.max_pool2d(
        net, ksize=3, strides=2, padding="SAME", name="initial_max_pool")

    for block_group in self._block_groups:
      net = block_group(net, is_training)

    if self._resnet_v2:
      net = self._final_batchnorm(net, is_training=is_training)
      net = tf.nn.relu(net)
    net = tf.reduce_mean(net, axis=[1, 2], name="final_avg_pool")
    return self._logits(net)


class ResNet50(ResNet):
  """ResNet50 module."""

  def __init__(self,
               num_classes: int,
               bn_config: Optional[Mapping[str, float]] = None,
               resnet_v2: bool = False,
               name: Optional[str] = None):
    """Constructs a ResNet model.

    Args:
      num_classes: The number of classes to classify the inputs into.
      bn_config: A dictionary of two elements, `decay_rate` and `eps` to be
        passed on to the `BatchNorm` layers.
      resnet_v2: Whether to use the v1 or v2 ResNet implementation. Defaults to
        False.
      name: Name of the module.
    """
    super().__init__([3, 4, 6, 3],
                     num_classes=num_classes,
                     bn_config=bn_config,
                     resnet_v2=resnet_v2,
                     name=name)
