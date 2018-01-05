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

"""Tests for `sonnet.python.modules.conv`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

# Dependency imports
from absl.testing import parameterized
import numpy as np
import sonnet as snt
import tensorflow as tf

from tensorflow.python.platform import test


def create_initializers(use_bias=True):
  if use_bias:
    return {"w": tf.truncated_normal_initializer(),
            "b": tf.constant_initializer()}
  else:
    return {"w": tf.truncated_normal_initializer()}


def create_custom_field_getter(conv_module, field_to_get):
  """Replace the tf.get_variable call for `field_to_get`."""
  def custom_getter(*args, **kwargs):  # pylint: disable=unused-argument
    return getattr(conv_module, field_to_get)
  return custom_getter


class Conv1DTestDataFormats(parameterized.TestCase, tf.test.TestCase):
  OUT_CHANNELS = 4
  KERNEL_SHAPE = 3
  INPUT_SHAPE = (2, 17, 18)

  def setUp(self):
    name = "{}.{}".format(type(self).__name__, self._testMethodName)
    if not test.is_gpu_available():
      self.skipTest("No GPU was detected, so {} will be skipped.".format(name))

  def checkEquality(self, o1, o2, w1=None, w2=None, atol=1e-5):
    with self.test_session(use_gpu=True, force_gpu=True):
      tf.global_variables_initializer().run()
      self.assertAllClose(o1.eval(), o2.eval(), atol=atol)

  @parameterized.named_parameters(
      ("WithBias_Stride1", True, 1), ("WithoutBias_Stride1", False, 1),
      ("WithBias_Stride2", True, 2), ("WithoutBias_Stride2", False, 2))
  def testConv1DDataFormats(self, use_bias, stride):
    """Check the module produces the same result for supported data formats."""
    func = functools.partial(
        snt.Conv1D,
        output_channels=self.OUT_CHANNELS,
        kernel_shape=self.KERNEL_SHAPE,
        use_bias=use_bias,
        stride=stride,
        initializers=create_initializers(use_bias))

    conv1 = func(name="NWC", data_format="NWC")
    x = tf.constant(np.random.random(self.INPUT_SHAPE).astype(np.float32))
    o1 = conv1(x)

    # We will force both modules to share the same weights by creating
    # a custom getter that returns the weights from the first conv module when
    # tf.get_variable is called.
    custom_getter = {"w": create_custom_field_getter(conv1, "w"),
                     "b": create_custom_field_getter(conv1, "b")}
    conv2 = func(name="NCW", data_format="NCW", custom_getter=custom_getter)
    x_transpose = tf.transpose(x, perm=(0, 2, 1))
    o2 = tf.transpose(conv2(x_transpose), perm=(0, 2, 1))

    self.checkEquality(o1, o2)

  @parameterized.named_parameters(("WithBias", True), ("WithoutBias", False))
  def testConv1DDataFormatsBatchNorm(self, use_bias):
    """Similar to `testConv1DDataFormats`, but this checks BatchNorm support."""

    def func(name, data_format, custom_getter=None):
      conv = snt.Conv1D(
          name=name,
          output_channels=self.OUT_CHANNELS,
          kernel_shape=self.KERNEL_SHAPE,
          use_bias=use_bias,
          initializers=create_initializers(use_bias),
          data_format=data_format,
          custom_getter=custom_getter)
      if data_format == "NWC":
        bn = snt.BatchNorm(scale=True, update_ops_collection=None)
      else:  # data_format = "NCW"
        bn = snt.BatchNorm(scale=True, update_ops_collection=None, axis=(0, 2))
      return snt.Sequential([conv, functools.partial(bn, is_training=True)])

    conv1 = func(name="NWC", data_format="NWC")
    x = tf.constant(np.random.random(self.INPUT_SHAPE).astype(np.float32))
    o1 = conv1(x)

    custom_getter = {"w": create_custom_field_getter(conv1.layers[0], "w"),
                     "b": create_custom_field_getter(conv1.layers[0], "b")}
    conv2 = func(name="NCW", data_format="NCW", custom_getter=custom_getter)
    x_transpose = tf.transpose(x, perm=(0, 2, 1))
    o2 = tf.transpose(conv2(x_transpose), perm=(0, 2, 1))

    self.checkEquality(o1, o2)


class Conv2DTestDataFormats(parameterized.TestCase, tf.test.TestCase):
  OUT_CHANNELS = 5
  KERNEL_SHAPE = 3
  INPUT_SHAPE = (2, 18, 19, 4)

  def setUp(self):
    name = "{}.{}".format(type(self).__name__, self._testMethodName)
    if not test.is_gpu_available():
      self.skipTest("No GPU was detected, so {} will be skipped.".format(name))

  def checkEquality(self, o1, o2, w1=None, w2=None, atol=1e-5):
    with self.test_session(use_gpu=True, force_gpu=True):
      tf.global_variables_initializer().run()
      self.assertAllClose(o1.eval(), o2.eval(), atol=atol)

  @parameterized.named_parameters(
      ("WithBias_Stride1", True, 1), ("WithoutBias_Stride1", False, 1),
      ("WithBias_Stride2", True, 2), ("WithoutBias_Stride2", False, 2))
  def testConv2DDataFormats(self, use_bias, stride):
    """Check the module produces the same result for supported data formats."""
    func = functools.partial(
        snt.Conv2D,
        output_channels=self.OUT_CHANNELS,
        kernel_shape=self.KERNEL_SHAPE,
        use_bias=use_bias,
        stride=stride,
        initializers=create_initializers(use_bias))

    conv1 = func(name="NHWC", data_format="NHWC")
    x = tf.constant(np.random.random(self.INPUT_SHAPE).astype(np.float32))
    o1 = conv1(x)

    # We will force both modules to share the same weights by creating
    # a custom getter that returns the weights from the first conv module when
    # tf.get_variable is called.
    custom_getter = {"w": create_custom_field_getter(conv1, "w"),
                     "b": create_custom_field_getter(conv1, "b")}
    conv2 = func(name="NCHW", data_format="NCHW", custom_getter=custom_getter)
    x_transpose = tf.transpose(x, perm=(0, 3, 1, 2))
    o2 = tf.transpose(conv2(x_transpose), perm=(0, 2, 3, 1))

    self.checkEquality(o1, o2)

  @parameterized.named_parameters(("WithBias", True), ("WithoutBias", False))
  def testConv2DDataFormatsBatchNorm(self, use_bias):
    """Similar to `testConv2DDataFormats`, but this checks BatchNorm support."""

    def func(name, data_format, custom_getter=None):
      conv = snt.Conv2D(
          name=name,
          output_channels=self.OUT_CHANNELS,
          kernel_shape=self.KERNEL_SHAPE,
          use_bias=use_bias,
          initializers=create_initializers(use_bias),
          data_format=data_format,
          custom_getter=custom_getter)
      if data_format == "NHWC":
        bn = snt.BatchNorm(scale=True, update_ops_collection=None)
      else:  # data_format = "NCHW"
        bn = snt.BatchNorm(scale=True, update_ops_collection=None, fused=True,
                           axis=(0, 2, 3))
      return snt.Sequential([conv, functools.partial(bn, is_training=True)])

    conv1 = func(name="NHWC", data_format="NHWC")
    x = tf.constant(np.random.random(self.INPUT_SHAPE).astype(np.float32))
    o1 = conv1(x)

    custom_getter = {"w": create_custom_field_getter(conv1.layers[0], "w"),
                     "b": create_custom_field_getter(conv1.layers[0], "b")}
    conv2 = func(name="NCHW", data_format="NCHW", custom_getter=custom_getter)
    x_transpose = tf.transpose(x, perm=(0, 3, 1, 2))
    o2 = tf.transpose(conv2(x_transpose), perm=(0, 2, 3, 1))

    self.checkEquality(o1, o2)


class Conv3DTestDataFormats(parameterized.TestCase, tf.test.TestCase):
  OUT_CHANNELS = 5
  KERNEL_SHAPE = 3
  INPUT_SHAPE = (2, 17, 18, 19, 4)

  def setUp(self):
    name = "{}.{}".format(type(self).__name__, self._testMethodName)
    if not test.is_gpu_available():
      self.skipTest("No GPU was detected, so {} will be skipped.".format(name))

  def checkEquality(self, o1, o2, w1=None, w2=None, atol=1e-5):
    with self.test_session(use_gpu=True, force_gpu=True):
      tf.global_variables_initializer().run()
      if w1 and w2:
        tf.logging.info("W1: %s, W2: %s", np.sum(w1.eval()), np.sum(w2.eval()))
      self.assertAllClose(o1.eval(), o2.eval(), atol=atol)

  @parameterized.named_parameters(
      ("WithBias_Stride1", True, 1), ("WithoutBias_Stride1", False, 1),
      ("WithBias_Stride2", True, 2), ("WithoutBias_Stride2", False, 2))
  def testConv3DDataFormats(self, use_bias, stride):
    """Check the module produces the same result for supported data formats."""
    func = functools.partial(
        snt.Conv3D,
        output_channels=self.OUT_CHANNELS,
        kernel_shape=self.KERNEL_SHAPE,
        use_bias=use_bias,
        stride=stride,
        initializers=create_initializers(use_bias))

    conv1 = func(name="NDHWC", data_format="NDHWC")
    x = tf.constant(np.random.random(self.INPUT_SHAPE).astype(np.float32))
    o1 = conv1(x)

    # We will force both modules to share the same weights by creating
    # a custom getter that returns the weights from the first conv module when
    # tf.get_variable is called.
    custom_getter = {"w": create_custom_field_getter(conv1, "w"),
                     "b": create_custom_field_getter(conv1, "b")}
    conv2 = func(name="NCDHW", data_format="NCDHW", custom_getter=custom_getter)
    x_transpose = tf.transpose(x, perm=(0, 4, 1, 2, 3))
    o2 = tf.transpose(conv2(x_transpose), perm=(0, 2, 3, 4, 1))

    self.checkEquality(o1, o2)

  @parameterized.named_parameters(("WithBias", True), ("WithoutBias", False))
  def testConv3DDataFormatsBatchNorm(self, use_bias):
    """Similar to `testConv3DDataFormats`, but this checks BatchNorm support."""

    def func(name, data_format, custom_getter=None):
      conv = snt.Conv3D(
          name=name,
          output_channels=self.OUT_CHANNELS,
          kernel_shape=self.KERNEL_SHAPE,
          use_bias=use_bias,
          initializers=create_initializers(use_bias),
          data_format=data_format,
          custom_getter=custom_getter)
      if data_format == "NDHWC":
        bn = snt.BatchNorm(scale=True, update_ops_collection=None)
      else:  # data_format = "NCDHW"
        bn = snt.BatchNorm(scale=True, update_ops_collection=None,
                           axis=(0, 2, 3, 4))
      return snt.Sequential([conv, functools.partial(bn, is_training=True)])

    conv1 = func(name="NDHWC", data_format="NDHWC")
    x = tf.constant(np.random.random(self.INPUT_SHAPE).astype(np.float32))
    o1 = conv1(x)

    custom_getter = {"w": create_custom_field_getter(conv1.layers[0], "w"),
                     "b": create_custom_field_getter(conv1.layers[0], "b")}
    conv2 = func(name="NCDHW", data_format="NCDHW", custom_getter=custom_getter)
    x_transpose = tf.transpose(x, perm=(0, 4, 1, 2, 3))
    o2 = tf.transpose(conv2(x_transpose), perm=(0, 2, 3, 4, 1))

    self.checkEquality(o1, o2)


if __name__ == "__main__":
  tf.test.main()
