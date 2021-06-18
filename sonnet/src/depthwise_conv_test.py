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
"""Tests for sonnet.v2.src.depthwise_conv."""

from absl.testing import parameterized

import numpy as np
from sonnet.src import depthwise_conv
from sonnet.src import initializers
from sonnet.src import test_utils
import tensorflow as tf


def create_constant_initializers(w, b, with_bias):
  if with_bias:
    return {
        "w_init": initializers.Constant(w),
        "b_init": initializers.Constant(b)
    }
  else:
    return {"w_init": initializers.Constant(w)}


class DepthwiseConvTest(test_utils.TestCase, parameterized.TestCase):

  def testInitializerKeysInvalidWithoutBias(self):
    with self.assertRaisesRegex(ValueError, "b_init must be None"):
      depthwise_conv.DepthwiseConv2D(
          channel_multiplier=1,
          kernel_shape=3,
          data_format="NHWC",
          with_bias=False,
          b_init=tf.zeros_initializer())

  @parameterized.parameters(tf.float32, tf.float64)
  def testDefaultInitializers(self, dtype):
    if "TPU" in self.device_types and dtype == tf.float64:
      self.skipTest("Double precision not supported on TPU.")

    conv1 = depthwise_conv.DepthwiseConv2D(
        kernel_shape=16, stride=1, padding="VALID", data_format="NHWC")

    out = conv1(tf.random.normal([8, 64, 64, 1], dtype=dtype))

    self.assertAllEqual(out.shape, [8, 49, 49, 1])
    self.assertEqual(out.dtype, dtype)

    # Note that for unit variance inputs the output is below unit variance
    # because of the use of the truncated normal initalizer
    err = 0.2 if self.primary_device == "TPU" else 0.1
    self.assertNear(out.numpy().std(), 0.87, err=err)

  @parameterized.named_parameters(("SamePaddingUseBias", True, "SAME"),
                                  ("SamePaddingNoBias", False, "SAME"),
                                  ("ValidPaddingNoBias", False, "VALID"),
                                  ("ValidPaddingUseBias", True, "VALID"))
  def testFunction(self, with_bias, padding):
    conv1 = depthwise_conv.DepthwiseConv2D(
        channel_multiplier=1,
        kernel_shape=3,
        stride=1,
        padding=padding,
        with_bias=with_bias,
        data_format="NHWC",
        **create_constant_initializers(1.0, 1.0, with_bias))
    conv2 = depthwise_conv.DepthwiseConv2D(
        channel_multiplier=1,
        kernel_shape=3,
        stride=1,
        padding=padding,
        with_bias=with_bias,
        data_format="NHWC",
        **create_constant_initializers(1.0, 1.0, with_bias))
    defun_conv = tf.function(conv2)

    iterations = 5

    for _ in range(iterations):
      x = tf.random.uniform([1, 5, 5, 1])
      y1 = conv1(x)
      y2 = defun_conv(x)

      self.assertAllClose(self.evaluate(y1), self.evaluate(y2), atol=1e-4)

  def testUnknownBatchSizeNHWC(self):
    x = tf.TensorSpec([None, 5, 5, 3], dtype=tf.float32)

    c = depthwise_conv.DepthwiseConv2D(
        channel_multiplier=1, kernel_shape=3, data_format="NHWC")
    defun_conv = tf.function(c).get_concrete_function(x)

    out1 = defun_conv(tf.ones([3, 5, 5, 3]))
    self.assertEqual(out1.shape, [3, 5, 5, 3])

    out2 = defun_conv(tf.ones([5, 5, 5, 3]))
    self.assertEqual(out2.shape, [5, 5, 5, 3])

  def testUnknownBatchSizeNCHW(self):
    if self.primary_device == "CPU":
      self.skipTest("NCHW not supported on CPU")

    x = tf.TensorSpec([None, 3, 5, 5], dtype=tf.float32)
    c = depthwise_conv.DepthwiseConv2D(
        channel_multiplier=1, kernel_shape=3, data_format="NCHW")
    defun_conv = tf.function(c).get_concrete_function(x)

    out1 = defun_conv(tf.ones([3, 3, 5, 5]))
    self.assertEqual(out1.shape, [3, 3, 5, 5])

    out2 = defun_conv(tf.ones([5, 3, 5, 5]))
    self.assertEqual(out2.shape, [5, 3, 5, 5])

  def testUnknownSpatialDims(self):
    x = tf.TensorSpec([3, None, None, 3], dtype=tf.float32)

    c = depthwise_conv.DepthwiseConv2D(
        channel_multiplier=1, kernel_shape=3, data_format="NHWC")
    defun_conv = tf.function(c).get_concrete_function(x)

    out = defun_conv(tf.ones([3, 5, 5, 3]))
    expected_out = c(tf.ones([3, 5, 5, 3]))
    self.assertEqual(out.shape, [3, 5, 5, 3])
    self.assertAllEqual(self.evaluate(out), self.evaluate(expected_out))

    out = defun_conv(tf.ones([3, 4, 4, 3]))
    expected_out = c(tf.ones([3, 4, 4, 3]))
    self.assertEqual(out.shape, [3, 4, 4, 3])
    self.assertAllEqual(self.evaluate(out), self.evaluate(expected_out))

  @parameterized.parameters(True, False)
  def testUnknownChannels(self, autograph):
    x = tf.TensorSpec([3, 3, 3, None], dtype=tf.float32)

    c = depthwise_conv.DepthwiseConv2D(
        channel_multiplier=1, kernel_shape=3, data_format="NHWC")
    defun_conv = tf.function(c, autograph=autograph)

    with self.assertRaisesRegex(ValueError,
                                "The number of input channels must be known"):
      defun_conv.get_concrete_function(x)

  @parameterized.named_parameters(("WithBias", True), ("WithoutBias", False))
  def testComputationSame(self, with_bias):
    conv1 = depthwise_conv.DepthwiseConv2D(
        channel_multiplier=1,
        kernel_shape=[3, 3],
        stride=1,
        padding="SAME",
        with_bias=with_bias,
        **create_constant_initializers(1.0, 1.0, with_bias))

    out = conv1(tf.ones([1, 5, 5, 1]))
    expected_out = np.array([[5, 7, 7, 7, 5], [7, 10, 10, 10, 7],
                             [7, 10, 10, 10, 7], [7, 10, 10, 10, 7],
                             [5, 7, 7, 7, 5]])
    if not with_bias:
      expected_out -= 1

    self.assertEqual(out.shape, [1, 5, 5, 1])
    self.assertAllClose(np.reshape(out.numpy(), [5, 5]), expected_out)

  @parameterized.named_parameters(("WithBias", True), ("WithoutBias", False))
  def testComputationValid(self, with_bias):
    conv1 = depthwise_conv.DepthwiseConv2D(
        channel_multiplier=1,
        kernel_shape=[3, 3],
        stride=1,
        padding="VALID",
        with_bias=with_bias,
        **create_constant_initializers(1.0, 1.0, with_bias))

    out = conv1(tf.ones([1, 5, 5, 1]))
    expected_out = np.array([[10, 10, 10], [10, 10, 10], [10, 10, 10]])
    if not with_bias:
      expected_out -= 1

    self.assertEqual(out.shape, [1, 3, 3, 1])
    self.assertAllClose(np.reshape(out.numpy(), [3, 3]), expected_out)

  @parameterized.named_parameters(("WithBias", True), ("WithoutBias", False))
  def testComputationValidMultiChannel(self, with_bias):
    conv1 = depthwise_conv.DepthwiseConv2D(
        channel_multiplier=1,
        kernel_shape=[3, 3],
        stride=1,
        padding="VALID",
        with_bias=with_bias,
        **create_constant_initializers(1.0, 1.0, with_bias))

    out = conv1(tf.ones([1, 5, 5, 3]))
    expected_out = np.array([[[10] * 3] * 3] * 3)
    if not with_bias:
      expected_out -= 1

    self.assertAllClose(np.reshape(out.numpy(), [3, 3, 3]), expected_out)

  @parameterized.named_parameters(("WithBias", True), ("WithoutBias", False))
  def testSharing(self, with_bias):
    """Sharing is working."""
    conv1 = depthwise_conv.DepthwiseConv2D(
        channel_multiplier=3,
        kernel_shape=3,
        stride=1,
        padding="SAME",
        with_bias=with_bias)

    x = np.random.randn(1, 5, 5, 1)
    x1 = tf.constant(x, dtype=np.float32)
    x2 = tf.constant(x, dtype=np.float32)

    self.assertAllClose(conv1(x1), conv1(x2))

    # Kernel shape was set to 3, which is expandeded to [3, 3, 3].
    # Input channels are 1, output channels := in_channels * multiplier.
    # multiplier is kernel_shape[2] == 3. So weight layout must be:
    # (3, 3, 1, 3).
    w = np.random.randn(3, 3, 1, 3)  # Now change the weights.
    conv1.w.assign(w)
    self.assertAllClose(conv1(x1), conv1(x2))


if __name__ == "__main__":
  tf.test.main()
