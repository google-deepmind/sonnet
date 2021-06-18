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
"""Tests for sonnet.v2.src.conv_transpose."""

import itertools

from absl.testing import parameterized
import numpy as np
from sonnet.src import conv_transpose
from sonnet.src import initializers as lib_initializers
from sonnet.src import test_utils
import tensorflow as tf


def create_constant_initializers(w, b, with_bias):
  if with_bias:
    return {
        "w_init": lib_initializers.Constant(w),
        "b_init": lib_initializers.Constant(b)
    }
  else:
    return {"w_init": lib_initializers.Constant(w)}


class ConvTransposeTest(test_utils.TestCase, parameterized.TestCase):

  @parameterized.parameters(0, 4)
  def testIncorrectN(self, n):
    with self.assertRaisesRegex(
        ValueError,
        "only support transpose convolution operations for num_spatial_dims"):
      conv_transpose.ConvNDTranspose(
          num_spatial_dims=n,
          output_channels=1,
          output_shape=None,
          kernel_shape=3,
          data_format="NHWC")

  def testIncorrectPadding(self):
    with self.assertRaisesRegex(
        TypeError,
        "ConvNDTranspose only takes string padding, please provide either"):
      conv_transpose.ConvNDTranspose(
          2, output_channels=1, kernel_shape=3, padding=None)

  def testBiasInitNoBias(self):
    with self.assertRaisesRegex(
        ValueError, "When not using a bias the b_init must be None."):
      conv_transpose.ConvNDTranspose(
          2, output_channels=1, kernel_shape=3, with_bias=False,
          b_init=lib_initializers.Ones(), data_format="NHWC")

  def testIncorrectOutputShape(self):
    c = conv_transpose.ConvNDTranspose(
        num_spatial_dims=2,
        output_channels=3,
        kernel_shape=2,
        output_shape=[1],
        data_format="NHWC")
    with self.assertRaisesRegex(
        ValueError, "The output_shape must be of length 2 but instead was 1."):
      c(tf.ones([3, 5, 5, 3]))

  @parameterized.parameters(*itertools.product(
      [True, False],  # with_bias
      ["SAME", "VALID"]))  # padding
  def testGraphConv(self, with_bias, padding):
    conv1 = conv_transpose.ConvNDTranspose(
        num_spatial_dims=2,
        output_channels=1,
        output_shape=None,
        kernel_shape=3,
        stride=1,
        padding=padding,
        with_bias=with_bias,
        data_format="NHWC",
        **create_constant_initializers(1.0, 1.0, with_bias))
    conv2 = conv_transpose.ConvNDTranspose(
        num_spatial_dims=2,
        output_channels=1,
        output_shape=None,
        kernel_shape=3,
        stride=1,
        padding=padding,
        with_bias=with_bias,
        data_format="NHWC",
        **create_constant_initializers(1.0, 1.0, with_bias))
    defun_conv = tf.function(conv2)

    iterations = 5

    for _ in range(iterations):
      x = tf.random.uniform([1, 3, 3, 1])
      y1 = conv1(x)
      y2 = defun_conv(x)

      self.assertAllClose(self.evaluate(y1), self.evaluate(y2), atol=1e-4)

  def testUnknownBatchSizeNHWC(self):
    x = tf.TensorSpec([None, 5, 5, 3], dtype=tf.float32)

    c = conv_transpose.ConvNDTranspose(
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

    c = conv_transpose.ConvNDTranspose(
        num_spatial_dims=2,
        output_channels=2,
        kernel_shape=3,
        data_format="NCHW")
    defun_conv = tf.function(c).get_concrete_function(x)

    out1 = defun_conv(tf.ones([3, 3, 5, 5]))
    self.assertEqual(out1.shape, [3, 2, 5, 5])

    out2 = defun_conv(tf.ones([5, 3, 5, 5]))
    self.assertEqual(out2.shape, [5, 2, 5, 5])

  def testUnknownShapeDims(self):
    x = tf.TensorSpec([3, None, None, 3], dtype=tf.float32)

    c = conv_transpose.ConvNDTranspose(
        num_spatial_dims=2,
        output_channels=2,
        kernel_shape=3,
        data_format="NHWC")
    defun_conv = tf.function(c).get_concrete_function(x)

    out1 = defun_conv(tf.ones([3, 5, 5, 3]))
    self.assertEqual(out1.shape, [3, 5, 5, 2])

    out1 = defun_conv(tf.ones([3, 3, 3, 3]))
    self.assertEqual(out1.shape, [3, 3, 3, 2])

  def testGivenOutputShape(self):
    c = conv_transpose.ConvNDTranspose(
        num_spatial_dims=2,
        output_channels=2,
        kernel_shape=3,
        output_shape=[5, 5],
        data_format="NHWC")

    out1 = c(tf.ones([3, 5, 5, 3]))
    self.assertEqual(out1.shape, [3, 5, 5, 2])

  @parameterized.parameters(True, False)
  def testUnknownChannels(self, autograph):
    x = tf.TensorSpec([3, 3, 3, None], dtype=tf.float32)

    c = conv_transpose.ConvNDTranspose(
        num_spatial_dims=2,
        output_channels=1,
        kernel_shape=3,
        data_format="NHWC")
    defun_conv = tf.function(c, autograph=autograph)

    with self.assertRaisesRegex(ValueError,
                                "The number of input channels must be known"):
      defun_conv.get_concrete_function(x)

  @parameterized.parameters(
      (1, (3,), 128, 5, "NWC"),
      (2, (4, 4), 64, 3, "NHWC"),
      (3, (4, 4, 4), 64, 3, "NDHWC"))
  def testInitializerVariance(self, num_spatial_dims, kernel_shape,
                              in_channels, output_channels, data_format):
    inputs = tf.random.uniform([16] + ([32] * num_spatial_dims) + [in_channels])

    c = conv_transpose.ConvNDTranspose(
        num_spatial_dims=num_spatial_dims,
        kernel_shape=kernel_shape,
        output_channels=output_channels,
        data_format=data_format)
    c(inputs)

    actual_std = c.w.numpy().std()
    expected_std = 1 / (np.sqrt(np.prod(kernel_shape + (in_channels,))))

    # This ratio of the error compared to the expected std might be somewhere
    # around 0.15 normally. We check it is not > 0.5, as that would indicate
    # something seriously wrong (ie the previous buggy initialization).
    rel_diff = np.abs(actual_std - expected_std) / expected_std
    self.assertLess(rel_diff, 0.5)


class Conv2DTransposeTest(test_utils.TestCase, parameterized.TestCase):

  @parameterized.parameters(True, False)
  def testComputationPaddingSame(self, with_bias):
    expected_out = [[4, 6, 4], [6, 9, 6], [4, 6, 4]]
    expected_out = np.asarray(expected_out, dtype=np.float32)
    if with_bias:
      expected_out += 1

    conv_transpose1 = conv_transpose.Conv2DTranspose(
        output_channels=1,
        output_shape=None,
        kernel_shape=3,
        stride=1,
        padding="SAME",
        with_bias=with_bias,
        **create_constant_initializers(1.0, 1.0, with_bias))

    out = conv_transpose1(tf.ones([1, 3, 3, 1], dtype=tf.float32))
    self.assertEqual(out.shape, [1, 3, 3, 1])
    out = tf.squeeze(out, axis=(0, 3))

    self.assertAllClose(self.evaluate(out), expected_out)

  @parameterized.parameters(True, False)
  def testComputationPaddingValid(self, with_bias):
    expected_out = [[1, 2, 3, 2, 1], [2, 4, 6, 4, 2], [3, 6, 9, 6, 3],
                    [2, 4, 6, 4, 2], [1, 2, 3, 2, 1]]
    expected_out = np.asarray(expected_out, dtype=np.float32)
    if with_bias:
      expected_out += 1

    conv1 = conv_transpose.Conv2DTranspose(
        output_channels=1,
        output_shape=None,
        kernel_shape=3,
        stride=1,
        padding="VALID",
        with_bias=with_bias,
        **create_constant_initializers(1.0, 1.0, with_bias))

    out = conv1(tf.ones([1, 3, 3, 1], dtype=tf.float32))
    self.assertEqual(out.shape, [1, 5, 5, 1])
    out = tf.squeeze(out, axis=(0, 3))

    self.assertAllClose(self.evaluate(out), expected_out)

  def testShapeDilated(self):
    if "CPU" == self.primary_device:
      self.skipTest("Not supported on CPU")
    conv1 = conv_transpose.Conv2DTranspose(
        output_channels=1,
        output_shape=None,
        kernel_shape=3,
        rate=2,
        padding="VALID")

    out = conv1(tf.ones([1, 3, 3, 1]))
    self.assertEqual(out.shape, [1, 7, 7, 1])


class Conv1DTransposeTest(test_utils.TestCase, parameterized.TestCase):

  @parameterized.parameters(True, False)
  def testComputationPaddingSame(self, with_bias):
    expected_out = [2, 3, 2]
    expected_out = np.asarray(expected_out, dtype=np.float32)
    if with_bias:
      expected_out += 1

    conv1 = conv_transpose.Conv1DTranspose(
        output_channels=1,
        output_shape=None,
        kernel_shape=3,
        stride=1,
        padding="SAME",
        with_bias=with_bias,
        **create_constant_initializers(1.0, 1.0, with_bias))

    out = conv1(tf.ones([1, 3, 1], dtype=tf.float32))
    self.assertEqual(out.shape, [1, 3, 1])
    out = tf.squeeze(out, axis=(0, 2))

    self.assertAllClose(self.evaluate(out), expected_out)

  @parameterized.parameters(True, False)
  def testComputationPaddingValid(self, with_bias):
    expected_out = [1, 2, 3, 2, 1]
    expected_out = np.asarray(expected_out, dtype=np.float32)
    if with_bias:
      expected_out += 1

    conv1 = conv_transpose.Conv1DTranspose(
        output_channels=1,
        output_shape=None,
        kernel_shape=3,
        stride=1,
        padding="VALID",
        with_bias=with_bias,
        **create_constant_initializers(1.0, 1.0, with_bias))

    out = conv1(tf.ones([1, 3, 1], dtype=tf.float32))
    self.assertEqual(out.shape, [1, 5, 1])
    out = tf.squeeze(out, axis=(0, 2))

    self.assertAllClose(self.evaluate(out), expected_out)


class Conv3DTransposeTest(test_utils.TestCase, parameterized.TestCase):

  @parameterized.parameters(True, False)
  def testComputationPaddingSame(self, with_bias):
    expected_out = np.asarray([
        8, 12, 8, 12, 18, 12, 8, 12, 8, 12, 18, 12, 18, 27, 18, 12, 18, 12, 8,
        12, 8, 12, 18, 12, 8, 12, 8
    ]).reshape((3, 3, 3))
    if with_bias:
      expected_out += 1

    conv_transpose1 = conv_transpose.Conv3DTranspose(
        output_channels=1,
        output_shape=None,
        kernel_shape=3,
        stride=1,
        padding="SAME",
        with_bias=with_bias,
        **create_constant_initializers(1.0, 1.0, with_bias))

    out = conv_transpose1(tf.ones([1, 3, 3, 3, 1], dtype=tf.float32))
    self.assertEqual(out.shape, [1, 3, 3, 3, 1])
    out = tf.squeeze(out, axis=(0, 4))

    self.assertAllClose(self.evaluate(out), expected_out)

  @parameterized.parameters(True, False)
  def testComputationPaddingValid(self, with_bias):
    expected_out = np.asarray([
        1, 2, 3, 2, 1, 2, 4, 6, 4, 2, 3, 6, 9, 6, 3, 2, 4, 6, 4, 2, 1, 2, 3, 2,
        1, 2, 4, 6, 4, 2, 4, 8, 12, 8, 4, 6, 12, 18, 12, 6, 4, 8, 12, 8, 4, 2,
        4, 6, 4, 2, 3, 6, 9, 6, 3, 6, 12, 18, 12, 6, 9, 18, 27, 18, 9, 6, 12,
        18, 12, 6, 3, 6, 9, 6, 3, 2, 4, 6, 4, 2, 4, 8, 12, 8, 4, 6, 12, 18, 12,
        6, 4, 8, 12, 8, 4, 2, 4, 6, 4, 2, 1, 2, 3, 2, 1, 2, 4, 6, 4, 2, 3, 6, 9,
        6, 3, 2, 4, 6, 4, 2, 1, 2, 3, 2, 1.
    ]).reshape((5, 5, 5))
    if with_bias:
      expected_out += 1

    conv1 = conv_transpose.Conv3DTranspose(
        output_channels=1,
        output_shape=None,
        kernel_shape=3,
        stride=1,
        padding="VALID",
        with_bias=with_bias,
        **create_constant_initializers(1.0, 1.0, with_bias))

    out = conv1(tf.ones([1, 3, 3, 3, 1], dtype=tf.float32))
    self.assertEqual(out.shape, [1, 5, 5, 5, 1])
    out = tf.squeeze(out, axis=(0, 4))

    self.assertAllClose(self.evaluate(out), expected_out)


if __name__ == "__main__":
  tf.test.main()
