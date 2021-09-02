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
"""Tests for sonnet.v2.src.conv."""

from absl.testing import parameterized
import numpy as np
from sonnet.src import conv
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


class ConvTest(test_utils.TestCase, parameterized.TestCase):

  def testPaddingFunctionReached(self):
    self.reached = False

    def padding_func(*unused_args):
      padding_func.called = True
      return [0, 0]

    conv1 = conv.ConvND(
        num_spatial_dims=2,
        output_channels=1,
        kernel_shape=3,
        stride=1,
        padding=padding_func,
        data_format="NHWC",
        **create_constant_initializers(1.0, 1.0, True))

    conv1(tf.ones([1, 5, 5, 1], dtype=tf.float32))

    self.assertEqual(conv1.conv_padding, "VALID")
    self.assertEqual(conv1.padding_func, padding_func)
    self.assertTrue(getattr(padding_func, "called", False))

  @parameterized.parameters(0, 4)
  def testIncorrectN(self, n):
    with self.assertRaisesRegex(
        ValueError,
        "We only support convoltion operations for num_spatial_dims=1, 2 or 3"):
      conv.ConvND(
          num_spatial_dims=n,
          output_channels=1,
          kernel_shape=3,
          data_format="NHWC")

  def testInitializerKeysInvalidWithoutBias(self):
    with self.assertRaisesRegex(ValueError, "b_init must be None"):
      conv.ConvND(
          num_spatial_dims=2,
          output_channels=1,
          kernel_shape=3,
          data_format="NHWC",
          with_bias=False,
          b_init=tf.zeros_initializer())

  def testIncorrectRankInput(self):
    c = conv.ConvND(
        num_spatial_dims=2,
        output_channels=1,
        kernel_shape=3,
        data_format="NHWC")
    with self.assertRaisesRegex(ValueError, "Shape .* must have rank 4"):
      c(tf.ones([2, 4, 4]))

  @parameterized.parameters(tf.float32, tf.float64)
  def testDefaultInitializers(self, dtype):
    if "TPU" in self.device_types and dtype == tf.float64:
      self.skipTest("Double precision not supported on TPU.")

    conv1 = conv.ConvND(
        num_spatial_dims=2,
        output_channels=1,
        kernel_shape=16,
        stride=1,
        padding="VALID",
        data_format="NHWC")

    out = conv1(tf.random.normal([8, 64, 64, 1], dtype=dtype))

    self.assertAllEqual(out.shape, [8, 49, 49, 1])
    self.assertEqual(out.dtype, dtype)

    # Note that for unit variance inputs the output is below unit variance
    # because of the use of the truncated normal initalizer
    err = 0.2 if self.primary_device == "TPU" else 0.1
    self.assertNear(out.numpy().std(), 0.87, err=err)

  @parameterized.named_parameters(
      ("SamePaddingUseBias", True, "SAME"),
      ("SamePaddingNoBias", False, "SAME"),
      ("samePaddingUseBias", True, "same"),
      ("samePaddingNoBias", False, "same"),
      ("ValidPaddingNoBias", False, "VALID"),
      ("ValidPaddingUseBias", True, "VALID"),
      ("validPaddingNoBias", False, "valid"),
      ("validPaddingUseBias", True, "valid"),
  )
  def testFunction(self, with_bias, padding):
    conv1 = conv.ConvND(
        num_spatial_dims=2,
        output_channels=1,
        kernel_shape=3,
        stride=1,
        padding=padding,
        with_bias=with_bias,
        data_format="NHWC",
        **create_constant_initializers(1.0, 1.0, with_bias))
    conv2 = conv.ConvND(
        num_spatial_dims=2,
        output_channels=1,
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

    c = conv.ConvND(
        num_spatial_dims=2,
        output_channels=2,
        kernel_shape=3,
        data_format="NHWC")
    defun_conv = tf.function(c).get_concrete_function(x)

    out1 = defun_conv(tf.ones([3, 5, 5, 3]))
    self.assertEqual(out1.shape, [3, 5, 5, 2])

    out2 = defun_conv(tf.ones([5, 5, 5, 3]))
    self.assertEqual(out2.shape, [5, 5, 5, 2])

  def testUnknownBatchSizeNCHW(self):
    if self.primary_device == "CPU":
      self.skipTest("NCHW not supported on CPU")

    x = tf.TensorSpec([None, 3, 5, 5], dtype=tf.float32)
    c = conv.ConvND(
        num_spatial_dims=2,
        output_channels=2,
        kernel_shape=3,
        data_format="NCHW")
    defun_conv = tf.function(c).get_concrete_function(x)

    out1 = defun_conv(tf.ones([3, 3, 5, 5]))
    self.assertEqual(out1.shape, [3, 2, 5, 5])

    out2 = defun_conv(tf.ones([5, 3, 5, 5]))
    self.assertEqual(out2.shape, [5, 2, 5, 5])

  @parameterized.parameters(True, False)
  def testUnknownChannels(self, autograph):
    x = tf.TensorSpec([3, 3, 3, None], dtype=tf.float32)

    c = conv.ConvND(
        num_spatial_dims=2,
        output_channels=1,
        kernel_shape=3,
        data_format="NHWC")
    defun_conv = tf.function(c, autograph=autograph)

    with self.assertRaisesRegex(ValueError,
                                "The number of input channels must be known"):
      defun_conv.get_concrete_function(x)

  def testUnknownSpatialDims(self):
    x = tf.TensorSpec([3, None, None, 3], dtype=tf.float32)

    c = conv.ConvND(
        num_spatial_dims=2,
        output_channels=1,
        kernel_shape=3,
        data_format="NHWC")
    defun_conv = tf.function(c).get_concrete_function(x)

    out = defun_conv(tf.ones([3, 5, 5, 3]))
    expected_out = c(tf.ones([3, 5, 5, 3]))
    self.assertEqual(out.shape, [3, 5, 5, 1])
    self.assertAllEqual(self.evaluate(out), self.evaluate(expected_out))

    out = defun_conv(tf.ones([3, 4, 4, 3]))
    expected_out = c(tf.ones([3, 4, 4, 3]))
    self.assertEqual(out.shape, [3, 4, 4, 1])
    self.assertAllEqual(self.evaluate(out), self.evaluate(expected_out))


class Conv2DTest(test_utils.TestCase, parameterized.TestCase):

  @parameterized.parameters(True, False)
  def testComputationPaddingSame(self, with_bias):
    expected_out = [[4, 6, 6, 6, 4], [6, 9, 9, 9, 6], [6, 9, 9, 9, 6],
                    [6, 9, 9, 9, 6], [4, 6, 6, 6, 4]]
    conv1 = conv.Conv2D(
        output_channels=1,
        kernel_shape=3,
        stride=1,
        padding="SAME",
        with_bias=with_bias,
        **create_constant_initializers(1.0, 1.0, with_bias))

    out = conv1(tf.ones([1, 5, 5, 1], dtype=tf.float32))
    self.assertEqual(out.shape, [1, 5, 5, 1])
    out = tf.squeeze(out, axis=(0, 3))

    expected_out = np.asarray(expected_out, dtype=np.float32)
    if with_bias:
      expected_out += 1

    self.assertAllClose(self.evaluate(out), expected_out)

  @parameterized.parameters(True, False)
  def testComputationPaddingValid(self, with_bias):
    expected_out = [[9, 9, 9], [9, 9, 9], [9, 9, 9]]
    conv1 = conv.Conv2D(
        output_channels=1,
        kernel_shape=3,
        stride=1,
        padding="VALID",
        with_bias=with_bias,
        **create_constant_initializers(1.0, 1.0, with_bias))

    out = conv1(tf.ones([1, 5, 5, 1], dtype=tf.float32))
    self.assertEqual(out.shape, [1, 3, 3, 1])
    out = tf.squeeze(out, axis=(0, 3))

    expected_out = np.asarray(expected_out, dtype=np.float32)
    if with_bias:
      expected_out += 1

    self.assertAllClose(self.evaluate(out), expected_out)


class Conv1DTest(test_utils.TestCase, parameterized.TestCase):

  @parameterized.parameters(True, False)
  def testComputationPaddingSame(self, with_bias):
    expected_out = [2, 3, 3, 3, 2]
    conv1 = conv.Conv1D(
        output_channels=1,
        kernel_shape=3,
        stride=1,
        padding="SAME",
        with_bias=with_bias,
        **create_constant_initializers(1.0, 1.0, with_bias))

    out = conv1(tf.ones([1, 5, 1], dtype=tf.float32))
    self.assertEqual(out.shape, [1, 5, 1])
    out = tf.squeeze(out, axis=(0, 2))

    expected_out = np.asarray(expected_out, dtype=np.float32)
    if with_bias:
      expected_out += 1

    self.assertAllClose(self.evaluate(out), expected_out)

  @parameterized.parameters(True, False)
  def testComputationPaddingValid(self, with_bias):
    expected_out = [3, 3, 3]
    expected_out = np.asarray(expected_out, dtype=np.float32)
    if with_bias:
      expected_out += 1

    conv1 = conv.Conv1D(
        output_channels=1,
        kernel_shape=3,
        stride=1,
        padding="VALID",
        with_bias=with_bias,
        **create_constant_initializers(1.0, 1.0, with_bias))

    out = conv1(tf.ones([1, 5, 1], dtype=tf.float32))
    self.assertEqual(out.shape, [1, 3, 1])
    out = tf.squeeze(out, axis=(0, 2))

    self.assertAllClose(self.evaluate(out), expected_out)


class Conv3DTest(test_utils.TestCase, parameterized.TestCase):

  @parameterized.parameters(True, False)
  def testComputationPaddingSame(self, with_bias):
    expected_out = np.asarray([
        9, 13, 13, 13, 9, 13, 19, 19, 19, 13, 13, 19, 19, 19, 13, 13, 19, 19,
        19, 13, 9, 13, 13, 13, 9, 13, 19, 19, 19, 13, 19, 28, 28, 28, 19, 19,
        28, 28, 28, 19, 19, 28, 28, 28, 19, 13, 19, 19, 19, 13, 13, 19, 19, 19,
        13, 19, 28, 28, 28, 19, 19, 28, 28, 28, 19, 19, 28, 28, 28, 19, 13, 19,
        19, 19, 13, 13, 19, 19, 19, 13, 19, 28, 28, 28, 19, 19, 28, 28, 28, 19,
        19, 28, 28, 28, 19, 13, 19, 19, 19, 13, 9, 13, 13, 13, 9, 13, 19, 19,
        19, 13, 13, 19, 19, 19, 13, 13, 19, 19, 19, 13, 9, 13, 13, 13, 9
    ]).reshape((5, 5, 5))
    if not with_bias:
      expected_out -= 1

    conv1 = conv.Conv3D(
        output_channels=1,
        kernel_shape=3,
        stride=1,
        padding="SAME",
        with_bias=with_bias,
        **create_constant_initializers(1.0, 1.0, with_bias))

    out = conv1(tf.ones([1, 5, 5, 5, 1], dtype=tf.float32))
    self.assertEqual(out.shape, [1, 5, 5, 5, 1])
    out = tf.squeeze(out, axis=(0, 4))

    self.assertAllClose(self.evaluate(out), expected_out)

  @parameterized.parameters(True, False)
  def testComputationPaddingValid(self, with_bias):
    expected_out = np.asarray([
        28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28,
        28, 28, 28, 28, 28, 28, 28, 28, 28
    ]).reshape((3, 3, 3))
    if not with_bias:
      expected_out -= 1

    conv1 = conv.Conv3D(
        output_channels=1,
        kernel_shape=3,
        stride=1,
        padding="VALID",
        with_bias=with_bias,
        **create_constant_initializers(1.0, 1.0, with_bias))

    out = conv1(tf.ones([1, 5, 5, 5, 1], dtype=tf.float32))
    self.assertEqual(out.shape, [1, 3, 3, 3, 1])
    out = tf.squeeze(out, axis=(0, 4))

    self.assertAllClose(self.evaluate(out), expected_out)


if __name__ == "__main__":
  tf.test.main()
