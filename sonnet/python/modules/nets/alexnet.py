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

# pylint: disable=line-too-long
"""Implementation of AlexNet as a Sonnet module.

`AlexNet` is a Sonnet module that implements two variants of
   'ImageNet Classification with Deep Convolutional Neural Networks'
    Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton, NIPS 2012
    http://papers.nips.cc/paper/4824-imagenet-classification-w

The two modes are FULL and MINI, corresponding to the full dual-gpu version and
a cut-down version that is able to run on Cifar10.

AlexNet is no longer state of the art and isn't considered a good starting point
for a vision network.
"""
# pylint: enable=line-too-long

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from sonnet.python.modules import base
from sonnet.python.modules import basic
from sonnet.python.modules import batch_norm
from sonnet.python.modules import conv
from sonnet.python.modules import util
import tensorflow as tf


class AlexNet(base.AbstractModule):
  """Implementation of AlexNet with full and mini versions.

  Based on:
    'ImageNet Classification with Deep Convolutional Neural Networks'
    Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton, NIPS 2012
    http://papers.nips.cc/paper/4824-imagenet-classification-w
  """

  FULL = "FULL"
  MINI = "MINI"

  POSSIBLE_INITIALIZER_KEYS = {"w", "b"}

  def __init__(self,
               mode,
               use_batch_norm=False,
               batch_norm_config=None,
               initializers=None,
               partitioners=None,
               regularizers=None,
               bn_on_fc_layers=True,
               custom_getter=None,
               name="alex_net"):
    """Constructs AlexNet.

    Args:
      mode: Construction mode of network: `AlexNet.FULL` or `AlexNet.MINI`.
      use_batch_norm: Whether to use batch normalization between the output of
          a layer and the activation function.
      batch_norm_config: Optional mapping of additional configuration for the
          `snt.BatchNorm` modules.
      initializers: Optional dict containing ops to initialize the filters (with
          key 'w') or biases (with key 'b'). The default initializers are
          truncated normal initializers, which are commonly used when the inputs
          are zero centered (see https://arxiv.org/pdf/1502.03167v3.pdf).
      partitioners: Optional dict containing partitioners for the filters
        (with key 'w') and the biases (with key 'b'). As a default, no
        partitioners are used.
      regularizers: Optional dict containing regularizers for the filters
        (with key 'w') and the biases (with key 'b'). As a default, no
        regularizers are used. A regularizer should be a function that takes
        a single `Tensor` as an input and returns a scalar `Tensor` output, e.g.
        the L1 and L2 regularizers in `tf.contrib.layers`.
      bn_on_fc_layers: If `use_batch_norm` is True, add batch normalization to
        the fully-connected layers. This is deprecated.
      custom_getter: Callable or dictionary of callables to use as
        custom getters inside the module. If a dictionary, the keys
        correspond to regexes to match variable names. See the `tf.get_variable`
        documentation for information about the custom_getter API.
      name: Name of the module.

    Raises:
      base.Error: If the given `mode` is not one of `AlexNet.FULL`,
        or `AlexNet.MINI`.
      KeyError: If `initializers`, `partitioners` or `regularizers` contains any
        keys other than 'w' or 'b'.
    """
    super(AlexNet, self).__init__(custom_getter=custom_getter, name=name)

    self._mode = mode
    self._use_batch_norm = use_batch_norm
    self._bn_on_fc_layers = bn_on_fc_layers

    if self._bn_on_fc_layers:
      tf.logging.warn("Using BatchNorm on the fully connected layers in "
                      "AlexNet is not recommended. 'bn_on_fc_layers' is a "
                      "deprecated option and will likely be removed.")

    self._batch_norm_config = batch_norm_config or {}

    if self._mode == self.FULL:
      # The full AlexNet, i.e. originally ran on two GPUs
      self._conv_layers = [
          (96, (11, 4), (3, 2)),
          (256, (5, 1), (3, 2)),
          (384, (3, 1), None),
          (384, (3, 1), None),
          (256, (3, 1), (3, 2)),
      ]

      self._fc_layers = [4096, 4096]
    elif self._mode == self.MINI:
      # A cut down version of the half net for testing with Cifar10
      self._conv_layers = [
          (48, (3, 1), (3, 1)),
          (128, (3, 1), (3, 1)),
          (192, (3, 1), None),
          (192, (3, 1), None),
          (128, (3, 1), (3, 1)),
      ]

      self._fc_layers = [1024, 1024]
    else:
      raise base.Error("AlexNet construction mode '{}' not recognised, "
                       "must be one of: '{}', '{}'".format(
                           mode, self.FULL, self.MINI))

    self._min_size = self._calc_min_size(self._conv_layers)
    self._conv_modules = []
    self._linear_modules = []

    # Keep old name for backwards compatibility

    self.possible_keys = self.POSSIBLE_INITIALIZER_KEYS

    self._initializers = util.check_initializers(
        initializers, self.POSSIBLE_INITIALIZER_KEYS)
    self._partitioners = util.check_partitioners(
        partitioners, self.POSSIBLE_INITIALIZER_KEYS)
    self._regularizers = util.check_regularizers(
        regularizers, self.POSSIBLE_INITIALIZER_KEYS)

  def _calc_min_size(self, conv_layers):
    """Calculates the minimum size of the input layer.

    Given a set of convolutional layers, calculate the minimum value of
    the `input_height` and `input_width`, i.e. such that the output has
    size 1x1. Assumes snt.VALID padding.

    Args:
      conv_layers: List of tuples `(output_channels, (kernel_size, stride),
        (pooling_size, pooling_stride))`

    Returns:
      Minimum value of input height and width.
    """
    input_size = 1

    for _, conv_params, max_pooling in reversed(conv_layers):
      if max_pooling is not None:
        kernel_size, stride = max_pooling
        input_size = input_size * stride + (kernel_size - stride)

      if conv_params is not None:
        kernel_size, stride = conv_params
        input_size = input_size * stride + (kernel_size - stride)

    return input_size

  def _build(self, inputs, keep_prob=None, is_training=None,
             test_local_stats=True):
    """Connects the AlexNet module into the graph.

    The is_training flag only controls the batch norm settings, if `False` it
    does not force no dropout by overriding any input `keep_prob`. To avoid any
    confusion this may cause, if `is_training=False` and `keep_prob` would cause
    dropout to be applied, an error is thrown.

    Args:
      inputs: A Tensor of size [batch_size, input_height, input_width,
        input_channels], representing a batch of input images.
      keep_prob: A scalar Tensor representing the dropout keep probability.
        When `is_training=False` this must be None or 1 to give no dropout.
      is_training: Boolean to indicate if we are currently training. Must be
          specified if batch normalization or dropout is used.
      test_local_stats: Boolean to indicate to `snt.BatchNorm` if batch
        normalization should  use local batch statistics at test time.
        By default `True`.

    Returns:
      A Tensor of size [batch_size, output_size], where `output_size` depends
      on the mode the network was constructed in.

    Raises:
      base.IncompatibleShapeError: If any of the input image dimensions
        (input_height, input_width) are too small for the given network mode.
      ValueError: If `keep_prob` is not None or 1 when `is_training=False`.
      ValueError: If `is_training` is not explicitly specified when using
        batch normalization.
    """
    # Check input shape
    if (self._use_batch_norm or keep_prob is not None) and is_training is None:
      raise ValueError("Boolean is_training flag must be explicitly specified "
                       "when using batch normalization or dropout.")

    input_shape = inputs.get_shape().as_list()
    if input_shape[1] < self._min_size or input_shape[2] < self._min_size:
      raise base.IncompatibleShapeError(
          "Image shape too small: ({:d}, {:d}) < {:d}".format(
              input_shape[1], input_shape[2], self._min_size))

    net = inputs

    # Check keep prob
    if keep_prob is not None:
      valid_inputs = tf.logical_or(is_training, tf.equal(keep_prob, 1.))
      keep_prob_check = tf.assert_equal(
          valid_inputs, True,
          message="Input `keep_prob` must be None or 1 if `is_training=False`.")
      with tf.control_dependencies([keep_prob_check]):
        net = tf.identity(net)

    for i, params in enumerate(self._conv_layers):
      output_channels, conv_params, max_pooling = params

      kernel_size, stride = conv_params

      conv_mod = conv.Conv2D(
          name="conv_{}".format(i),
          output_channels=output_channels,
          kernel_shape=kernel_size,
          stride=stride,
          padding=conv.VALID,
          initializers=self._initializers,
          partitioners=self._partitioners,
          regularizers=self._regularizers)

      if not self.is_connected:
        self._conv_modules.append(conv_mod)

      net = conv_mod(net)

      if self._use_batch_norm:
        bn = batch_norm.BatchNorm(**self._batch_norm_config)
        net = bn(net, is_training, test_local_stats)

      net = tf.nn.relu(net)

      if max_pooling is not None:
        pooling_kernel_size, pooling_stride = max_pooling
        net = tf.nn.max_pool(
            net,
            ksize=[1, pooling_kernel_size, pooling_kernel_size, 1],
            strides=[1, pooling_stride, pooling_stride, 1],
            padding=conv.VALID)

    net = basic.BatchFlatten(name="flatten")(net)

    for i, output_size in enumerate(self._fc_layers):
      linear_mod = basic.Linear(
          name="fc_{}".format(i),
          output_size=output_size,
          initializers=self._initializers,
          partitioners=self._partitioners)

      if not self.is_connected:
        self._linear_modules.append(linear_mod)

      net = linear_mod(net)

      if self._use_batch_norm and self._bn_on_fc_layers:
        bn = batch_norm.BatchNorm(**self._batch_norm_config)
        net = bn(net, is_training, test_local_stats)

      net = tf.nn.relu(net)

      if keep_prob is not None:
        net = tf.nn.dropout(net, keep_prob=keep_prob)

    return net

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
  def min_input_size(self):
    """Returns integer specifying the minimum width and height for the input.

    Note that the input can be non-square, but both the width and height must
    be >= this number in size.

    Returns:
      The minimum size as an integer.
    """
    return self._min_size

  @property
  def conv_modules(self):
    """Returns list containing convolutional modules of network.

    Returns:
      A list containing the Conv2D modules.
    """
    self._ensure_is_connected()

    return self._conv_modules

  @property
  def linear_modules(self):
    """Returns list containing linear modules of network.

    Returns:
      A list containing the Linear modules.
    """
    self._ensure_is_connected()

    return self._linear_modules


class AlexNetFull(AlexNet):
  """AlexNet constructed in the 'FULL' mode."""

  def __init__(self,
               use_batch_norm=False,
               batch_norm_config=None,
               initializers=None,
               partitioners=None,
               regularizers=None,
               custom_getter=None,
               name="alex_net_full"):
    """Constructs AlexNet.

    Args:
      use_batch_norm: Whether to use batch normalization between the output of
          a layer and the activation function.
      batch_norm_config: Optional mapping of additional configuration for the
          `snt.BatchNorm` modules.
      initializers: Optional dict containing ops to initialize the filters (with
          key 'w') or biases (with key 'b'). The default initializers are
          truncated normal initializers, which are commonly used when the inputs
          are zero centered (see https://arxiv.org/pdf/1502.03167v3.pdf).
      partitioners: Optional dict containing partitioners for the filters
        (with key 'w') and the biases (with key 'b'). As a default, no
        partitioners are used.
      regularizers: Optional dict containing regularizers for the filters
        (with key 'w') and the biases (with key 'b'). As a default, no
        regularizers are used. A regularizer should be a function that takes
        a single `Tensor` as an input and returns a scalar `Tensor` output, e.g.
        the L1 and L2 regularizers in `tf.contrib.layers`.
      custom_getter: Callable or dictionary of callables to use as
        custom getters inside the module. If a dictionary, the keys
        correspond to regexes to match variable names. See the `tf.get_variable`
        documentation for information about the custom_getter API.
      name: Name of the module.

    Raises:
      KeyError: If `initializers`, `partitioners` or `regularizers` contains any
        keys other than 'w' or 'b'.
    """
    super(AlexNetFull, self).__init__(
        mode=self.FULL,
        use_batch_norm=use_batch_norm,
        batch_norm_config=batch_norm_config,
        initializers=initializers,
        partitioners=partitioners,
        regularizers=regularizers,
        bn_on_fc_layers=False,
        custom_getter=custom_getter,
        name=name)


class AlexNetMini(AlexNet):
  """AlexNet constructed in the 'MINI' mode."""

  def __init__(self,
               use_batch_norm=False,
               batch_norm_config=None,
               initializers=None,
               partitioners=None,
               regularizers=None,
               custom_getter=None,
               name="alex_net_mini"):
    """Constructs AlexNet.

    Args:
      use_batch_norm: Whether to use batch normalization between the output of
          a layer and the activation function.
      batch_norm_config: Optional mapping of additional configuration for the
          `snt.BatchNorm` modules.
      initializers: Optional dict containing ops to initialize the filters (with
          key 'w') or biases (with key 'b'). The default initializers are
          truncated normal initializers, which are commonly used when the inputs
          are zero centered (see https://arxiv.org/pdf/1502.03167v3.pdf).
      partitioners: Optional dict containing partitioners for the filters
        (with key 'w') and the biases (with key 'b'). As a default, no
        partitioners are used.
      regularizers: Optional dict containing regularizers for the filters
        (with key 'w') and the biases (with key 'b'). As a default, no
        regularizers are used. A regularizer should be a function that takes
        a single `Tensor` as an input and returns a scalar `Tensor` output, e.g.
        the L1 and L2 regularizers in `tf.contrib.layers`.
      custom_getter: Callable or dictionary of callables to use as
        custom getters inside the module. If a dictionary, the keys
        correspond to regexes to match variable names. See the `tf.get_variable`
        documentation for information about the custom_getter API.
      name: Name of the module.

    Raises:
      KeyError: If `initializers`, `partitioners` or `regularizers` contains any
        keys other than 'w' or 'b'.
    """
    super(AlexNetMini, self).__init__(
        mode=self.MINI,
        use_batch_norm=use_batch_norm,
        batch_norm_config=batch_norm_config,
        initializers=initializers,
        partitioners=partitioners,
        regularizers=regularizers,
        bn_on_fc_layers=False,
        custom_getter=custom_getter,
        name=name)
