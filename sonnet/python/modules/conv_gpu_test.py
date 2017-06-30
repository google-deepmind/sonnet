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
import numpy as np
import sonnet as snt
from sonnet.testing import parameterized
import tensorflow as tf

from tensorflow.python.platform import test


def create_constant_initializers(w, b, use_bias=True):
  if use_bias:
    return {"w": tf.constant_initializer(w), "b": tf.constant_initializer(b)}
  else:
    return {"w": tf.constant_initializer(w)}


class ConvTestDataFormats(parameterized.ParameterizedTestCase,
                          tf.test.TestCase):
  OUT_CHANNELS = 5
  KERNEL_SHAPE = 3
  INPUT_SHAPE = (2, 19, 19, 4)

  def setUp(self):
    name = "{}.{}".format(type(self).__name__, self._testMethodName)
    if not test.is_gpu_available():
      self.skipTest("No GPU was detected, so {} will be skipped.".format(name))

  def helperDataFormats(self, func, x, use_bias, atol=1e-5):
    """Test whether the result for different Data Formats is the same."""
    mod1 = func(name="default")
    mod2 = func(name="NCHW_conv", data_format="NCHW")
    x_transpose = tf.transpose(x, perm=(0, 3, 1, 2))
    o1 = mod1(x)
    o2 = tf.transpose(mod2(x_transpose), perm=(0, 2, 3, 1))
    with self.test_session(use_gpu=True, force_gpu=True):
      tf.global_variables_initializer().run()
      self.assertAllClose(o1.eval(), o2.eval(), atol=atol)

  @parameterized.NamedParameters(("WithBias", True), ("WithoutBias", False))
  def testConv2DDataFormats(self, use_bias):
    """Test data formats for Conv2D."""
    func = functools.partial(
        snt.Conv2D,
        output_channels=self.OUT_CHANNELS,
        kernel_shape=self.KERNEL_SHAPE,
        use_bias=use_bias,
        initializers=create_constant_initializers(1.0, 1.0, use_bias))
    x = tf.constant(np.random.random(self.INPUT_SHAPE).astype(np.float32))
    self.helperDataFormats(func, x, use_bias)

  @parameterized.NamedParameters(("WithBias", True), ("WithoutBias", False))
  def testConv2DTransposeDataFormats(self, use_bias):
    """Test data formats for Conv2DTranspose."""

    mb, h, w, c = self.INPUT_SHAPE

    def func(name, data_format="NHWC"):
      shape = self.INPUT_SHAPE if data_format == "NHWC" else (mb, c, h, w)
      temp_input = tf.constant(0.0, dtype=tf.float32, shape=shape)
      mod = snt.Conv2D(
          name=name,
          output_channels=self.OUT_CHANNELS,
          kernel_shape=self.KERNEL_SHAPE,
          use_bias=use_bias,
          initializers=create_constant_initializers(1.0, 1.0, use_bias),
          data_format=data_format)
      _ = mod(temp_input)
      return mod.transpose(name=name + "Trans")

    shape = (mb, h, w, self.OUT_CHANNELS)
    x = tf.constant(np.random.random(shape).astype(np.float32))
    self.helperDataFormats(func, x, use_bias)

  @parameterized.NamedParameters(("WithBias", True), ("WithoutBias", False))
  def testConv2DDataFormatsBatchNorm(self, use_bias):
    """Tests data formats for the convolutions with batch normalization."""

    def func(name, data_format="NHWC"):
      conv = snt.Conv2D(
          name=name,
          output_channels=self.OUT_CHANNELS,
          kernel_shape=self.KERNEL_SHAPE,
          use_bias=use_bias,
          initializers=create_constant_initializers(1.0, 1.0, use_bias),
          data_format=data_format)
      if data_format == "NHWC":
        bn = snt.BatchNorm(scale=True, update_ops_collection=None)
      else:
        bn = snt.BatchNorm(scale=True, update_ops_collection=None, fused=True,
                           axis=(0, 2, 3))
      return snt.Sequential([conv, functools.partial(bn, is_training=True)])
    x = tf.constant(np.random.random(self.INPUT_SHAPE).astype(np.float32))
    self.helperDataFormats(func, x, use_bias)


if __name__ == "__main__":
  tf.test.main()
