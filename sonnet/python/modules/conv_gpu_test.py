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

import collections
import functools

# Dependency imports
from absl.testing import parameterized
import numpy as np
import sonnet as snt
import tensorflow as tf

from tensorflow.python.platform import test

Conv1DInput = collections.namedtuple(
    "Conv1DInput", ["input_batch", "input_width", "input_channels"])

Conv2DInput = collections.namedtuple(
    "Conv2DInput", ["input_batch", "input_height", "input_width",
                    "input_channels"])

Conv3DInput = collections.namedtuple(
    "Conv3DInput", ["input_batch", "input_depth", "input_height", "input_width",
                    "input_channels"])


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
  INPUT_SHAPE = Conv1DInput(2, 17, 18)

  def setUp(self):
    super(Conv1DTestDataFormats, self).setUp()
    name = "{}.{}".format(type(self).__name__, self._testMethodName)
    if not test.is_gpu_available():
      self.skipTest("No GPU was detected, so {} will be skipped.".format(name))

  def checkEquality(self, o1, o2, atol=1e-5):
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

    conv_nwc = func(name="NWC", data_format="NWC")
    x = tf.constant(np.random.random(self.INPUT_SHAPE).astype(np.float32))
    result_nwc = conv_nwc(x)

    # We will force both modules to share the same weights by creating
    # a custom getter that returns the weights from the first conv module when
    # tf.get_variable is called.
    custom_getter = {"w": create_custom_field_getter(conv_nwc, "w"),
                     "b": create_custom_field_getter(conv_nwc, "b")}
    conv_nwc = func(name="NCW", data_format="NCW", custom_getter=custom_getter)
    x_transpose = tf.transpose(x, perm=(0, 2, 1))
    result_ncw = tf.transpose(conv_nwc(x_transpose), perm=(0, 2, 1))

    self.checkEquality(result_nwc, result_ncw)

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
        batch_norm = snt.BatchNorm(scale=True, update_ops_collection=None)
      else:  # data_format = "NCW"
        batch_norm = snt.BatchNorm(scale=True, update_ops_collection=None,
                                   axis=(0, 2))
      return snt.Sequential([conv,
                             functools.partial(batch_norm, is_training=True)])

    seq_nwc = func(name="NWC", data_format="NWC")
    x = tf.constant(np.random.random(self.INPUT_SHAPE).astype(np.float32))
    result_nwc = seq_nwc(x)

    custom_getter = {"w": create_custom_field_getter(seq_nwc.layers[0], "w"),
                     "b": create_custom_field_getter(seq_nwc.layers[0], "b")}
    seq_ncw = func(name="NCW", data_format="NCW", custom_getter=custom_getter)
    x_transpose = tf.transpose(x, perm=(0, 2, 1))
    result_ncw = tf.transpose(seq_ncw(x_transpose), perm=(0, 2, 1))

    self.checkEquality(result_nwc, result_ncw)


class CausalConv1DTestDataFormats(parameterized.TestCase, tf.test.TestCase):
  OUT_CHANNELS = 4
  KERNEL_SHAPE = 3
  INPUT_SHAPE = Conv1DInput(2, 17, 18)

  def setUp(self):
    super(CausalConv1DTestDataFormats, self).setUp()
    name = "{}.{}".format(type(self).__name__, self._testMethodName)
    if not test.is_gpu_available():
      self.skipTest("No GPU was detected, so {} will be skipped.".format(name))

  def checkEquality(self, o1, o2, atol=1e-5):
    with self.test_session(use_gpu=True, force_gpu=True):
      tf.global_variables_initializer().run()
      self.assertAllClose(o1.eval(), o2.eval(), atol=atol)

  @parameterized.named_parameters(
      ("WithBias_Stride1", True, 1), ("WithoutBias_Stride1", False, 1),
      ("WithBias_Stride2", True, 2), ("WithoutBias_Stride2", False, 2))
  def testCausalConv1DDataFormats(self, use_bias, stride):
    """Check the module produces the same result for supported data formats."""
    func = functools.partial(
        snt.CausalConv1D,
        output_channels=self.OUT_CHANNELS,
        kernel_shape=self.KERNEL_SHAPE,
        use_bias=use_bias,
        stride=stride,
        initializers=create_initializers(use_bias))

    conv_nwc = func(name="NWC", data_format="NWC")
    x = tf.constant(np.random.random(self.INPUT_SHAPE).astype(np.float32))
    result_nwc = conv_nwc(x)

    # We will force both modules to share the same weights by creating
    # a custom getter that returns the weights from the first conv module when
    # tf.get_variable is called.
    custom_getter = {"w": create_custom_field_getter(conv_nwc, "w"),
                     "b": create_custom_field_getter(conv_nwc, "b")}
    conv_ncw = func(name="NCW", data_format="NCW", custom_getter=custom_getter)
    x_transpose = tf.transpose(x, perm=(0, 2, 1))
    result_ncw = tf.transpose(conv_ncw(x_transpose), perm=(0, 2, 1))

    self.checkEquality(result_nwc, result_ncw)

  @parameterized.named_parameters(("WithBias", True), ("WithoutBias", False))
  def testCausalConv1DDataFormatsBatchNorm(self, use_bias):
    """Similar to `testCausalConv1DDataFormats`. Checks BatchNorm support."""

    def func(name, data_format, custom_getter=None):
      conv = snt.CausalConv1D(
          name=name,
          output_channels=self.OUT_CHANNELS,
          kernel_shape=self.KERNEL_SHAPE,
          use_bias=use_bias,
          initializers=create_initializers(use_bias),
          data_format=data_format,
          custom_getter=custom_getter)
      if data_format == "NWC":
        batch_norm = snt.BatchNorm(scale=True, update_ops_collection=None)
      else:  # data_format == "NCW"
        batch_norm = snt.BatchNorm(scale=True, update_ops_collection=None,
                                   axis=(0, 2))
      return snt.Sequential([conv,
                             functools.partial(batch_norm, is_training=True)])

    seq_nwc = func(name="NWC", data_format="NWC")
    x = tf.constant(np.random.random(self.INPUT_SHAPE).astype(np.float32))
    result_nwc = seq_nwc(x)

    custom_getter = {"w": create_custom_field_getter(seq_nwc.layers[0], "w"),
                     "b": create_custom_field_getter(seq_nwc.layers[0], "b")}
    seq_ncw = func(name="NCW", data_format="NCW", custom_getter=custom_getter)
    x_transpose = tf.transpose(x, perm=(0, 2, 1))
    result_ncw = tf.transpose(seq_ncw(x_transpose), perm=(0, 2, 1))

    self.checkEquality(result_nwc, result_ncw)


class Conv2DTestDataFormats(parameterized.TestCase, tf.test.TestCase):
  OUT_CHANNELS = 5
  KERNEL_SHAPE = 3
  INPUT_SHAPE = Conv2DInput(2, 18, 19, 4)

  def setUp(self):
    super(Conv2DTestDataFormats, self).setUp()
    name = "{}.{}".format(type(self).__name__, self._testMethodName)
    if not test.is_gpu_available():
      self.skipTest("No GPU was detected, so {} will be skipped.".format(name))

  def checkEquality(self, o1, o2, atol=1e-5):
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

    conv_nhwc = func(name="NHWC", data_format="NHWC")
    x = tf.constant(np.random.random(self.INPUT_SHAPE).astype(np.float32))
    result_nhwc = conv_nhwc(x)

    # We will force both modules to share the same weights by creating
    # a custom getter that returns the weights from the first conv module when
    # tf.get_variable is called.
    custom_getter = {"w": create_custom_field_getter(conv_nhwc, "w"),
                     "b": create_custom_field_getter(conv_nhwc, "b")}
    conv_nchw = func(name="NCHW", data_format="NCHW",
                     custom_getter=custom_getter)
    x_transpose = tf.transpose(x, perm=(0, 3, 1, 2))
    result_nchw = tf.transpose(conv_nchw(x_transpose), perm=(0, 2, 3, 1))

    self.checkEquality(result_nhwc, result_nchw)

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
        batch_norm = snt.BatchNorm(scale=True, update_ops_collection=None)
      else:  # data_format = "NCHW"
        batch_norm = snt.BatchNorm(scale=True, update_ops_collection=None,
                                   fused=True, axis=(0, 2, 3))
      return snt.Sequential([conv,
                             functools.partial(batch_norm, is_training=True)])

    seq_nhwc = func(name="NHWC", data_format="NHWC")
    x = tf.constant(np.random.random(self.INPUT_SHAPE).astype(np.float32))
    result_nhwc = seq_nhwc(x)

    custom_getter = {"w": create_custom_field_getter(seq_nhwc.layers[0], "w"),
                     "b": create_custom_field_getter(seq_nhwc.layers[0], "b")}
    seq_nchw = func(name="NCHW", data_format="NCHW",
                    custom_getter=custom_getter)
    x_transpose = tf.transpose(x, perm=(0, 3, 1, 2))
    result_nchw = tf.transpose(seq_nchw(x_transpose), perm=(0, 2, 3, 1))

    self.checkEquality(result_nhwc, result_nchw)


class Conv3DTestDataFormats(parameterized.TestCase, tf.test.TestCase):
  OUT_CHANNELS = 5
  KERNEL_SHAPE = 3
  INPUT_SHAPE = Conv3DInput(2, 17, 18, 19, 4)

  def setUp(self):
    super(Conv3DTestDataFormats, self).setUp()
    name = "{}.{}".format(type(self).__name__, self._testMethodName)
    if not test.is_gpu_available():
      self.skipTest("No GPU was detected, so {} will be skipped.".format(name))

  def checkEquality(self, o1, o2, atol=1e-5):
    with self.test_session(use_gpu=True, force_gpu=True):
      tf.global_variables_initializer().run()
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

    conv_ndhwc = func(name="NDHWC", data_format="NDHWC")
    x = tf.constant(np.random.random(self.INPUT_SHAPE).astype(np.float32))
    result_ndhwc = conv_ndhwc(x)

    # We will force both modules to share the same weights by creating
    # a custom getter that returns the weights from the first conv module when
    # tf.get_variable is called.
    custom_getter = {"w": create_custom_field_getter(conv_ndhwc, "w"),
                     "b": create_custom_field_getter(conv_ndhwc, "b")}
    conv_ncdhw = func(name="NCDHW", data_format="NCDHW",
                      custom_getter=custom_getter)
    x_transpose = tf.transpose(x, perm=(0, 4, 1, 2, 3))
    result_ncdhw = tf.transpose(conv_ncdhw(x_transpose), perm=(0, 2, 3, 4, 1))

    self.checkEquality(result_ndhwc, result_ncdhw)

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
        batch_norm = snt.BatchNorm(scale=True, update_ops_collection=None)
      else:  # data_format = "NCDHW"
        batch_norm = snt.BatchNorm(scale=True, update_ops_collection=None,
                                   axis=(0, 2, 3, 4))
      return snt.Sequential([conv,
                             functools.partial(batch_norm, is_training=True)])

    seq_ndhwc = func(name="NDHWC", data_format="NDHWC")
    x = tf.constant(np.random.random(self.INPUT_SHAPE).astype(np.float32))
    result_ndhwc = seq_ndhwc(x)

    custom_getter = {"w": create_custom_field_getter(seq_ndhwc.layers[0], "w"),
                     "b": create_custom_field_getter(seq_ndhwc.layers[0], "b")}
    seq_ncdhw = func(name="NCDHW", data_format="NCDHW",
                     custom_getter=custom_getter)
    x_transpose = tf.transpose(x, perm=(0, 4, 1, 2, 3))
    result_ncdhw = tf.transpose(seq_ncdhw(x_transpose), perm=(0, 2, 3, 4, 1))

    self.checkEquality(result_ndhwc, result_ncdhw)


class Conv1DTransposeTestDataFormats(parameterized.TestCase, tf.test.TestCase):
  OUT_CHANNELS = 5
  KERNEL_SHAPE = 3
  INPUT_SHAPE = Conv1DInput(2, 17, 4)

  def setUp(self):
    super(Conv1DTransposeTestDataFormats, self).setUp()
    name = "{}.{}".format(type(self).__name__, self._testMethodName)
    if not test.is_gpu_available():
      self.skipTest("No GPU was detected, so {} will be skipped.".format(name))

  def checkEquality(self, o1, o2, atol=1e-5):
    with self.test_session(use_gpu=True, force_gpu=True):
      tf.global_variables_initializer().run()
      self.assertAllClose(o1.eval(), o2.eval(), atol=atol)

  @parameterized.named_parameters(
      ("WithBias_Stride1", True, 1), ("WithoutBias_Stride1", False, 1),
      ("WithBias_Stride2", True, 2), ("WithoutBias_Stride2", False, 2))
  def testConv1DTransposeDataFormats(self, use_bias, stride):
    """Check the module produces the same result for supported data formats."""
    input_shape = (self.INPUT_SHAPE.input_batch,
                   int(np.ceil(self.INPUT_SHAPE.input_width / stride)),
                   self.INPUT_SHAPE.input_channels)

    func = functools.partial(
        snt.Conv1DTranspose,
        output_channels=self.OUT_CHANNELS,
        kernel_shape=self.KERNEL_SHAPE,
        output_shape=(self.INPUT_SHAPE.input_width,),
        use_bias=use_bias,
        stride=stride,
        initializers=create_initializers(use_bias))

    conv_nwc = func(name="NWC", data_format="NWC")
    x = tf.constant(np.random.random(input_shape).astype(np.float32))
    result_nwc = conv_nwc(x)

    # We will force both modules to share the same weights by creating
    # a custom getter that returns the weights from the first conv module when
    # tf.get_variable is called.
    custom_getter = {"w": create_custom_field_getter(conv_nwc, "w"),
                     "b": create_custom_field_getter(conv_nwc, "b")}
    conv_ncw = func(name="NCW", data_format="NCW", custom_getter=custom_getter)
    x_transpose = tf.transpose(x, perm=(0, 2, 1))
    result_ncw = tf.transpose(conv_ncw(x_transpose), perm=(0, 2, 1))

    self.checkEquality(result_nwc, result_ncw)

  @parameterized.named_parameters(("WithBias", True), ("WithoutBias", False))
  def testConv1DTransposeDataFormatsBatchNorm(self, use_bias):
    """Like `testConv1DTransposeDataFormats` but checks BatchNorm support."""

    def func(name, data_format, custom_getter=None):
      conv = snt.Conv1DTranspose(
          name=name,
          output_channels=self.OUT_CHANNELS,
          kernel_shape=self.KERNEL_SHAPE,
          output_shape=(self.INPUT_SHAPE.input_width,),
          use_bias=use_bias,
          initializers=create_initializers(use_bias),
          data_format=data_format,
          custom_getter=custom_getter)
      if data_format == "NWC":
        batch_norm = snt.BatchNorm(scale=True, update_ops_collection=None)
      else:  # data_format == "NCW"
        batch_norm = snt.BatchNorm(scale=True, update_ops_collection=None,
                                   axis=(0, 2))
      return snt.Sequential([conv,
                             functools.partial(batch_norm, is_training=True)])

    seq_nwc = func(name="NWC", data_format="NWC")
    x = tf.constant(np.random.random(self.INPUT_SHAPE).astype(np.float32))
    result_nwc = seq_nwc(x)

    custom_getter = {"w": create_custom_field_getter(seq_nwc.layers[0], "w"),
                     "b": create_custom_field_getter(seq_nwc.layers[0], "b")}
    seq_ncw = func(name="NCW", data_format="NCW", custom_getter=custom_getter)
    x_transpose = tf.transpose(x, perm=(0, 2, 1))
    result_ncw = tf.transpose(seq_ncw(x_transpose), perm=(0, 2, 1))

    self.checkEquality(result_nwc, result_ncw)


class Conv2DTransposeTestDataFormats(parameterized.TestCase, tf.test.TestCase):
  OUT_CHANNELS = 5
  KERNEL_SHAPE = 3
  INPUT_SHAPE = Conv2DInput(2, 18, 19, 4)

  def setUp(self):
    super(Conv2DTransposeTestDataFormats, self).setUp()
    name = "{}.{}".format(type(self).__name__, self._testMethodName)
    if not test.is_gpu_available():
      self.skipTest("No GPU was detected, so {} will be skipped.".format(name))

  def checkEquality(self, o1, o2, atol=1e-5):
    with self.test_session(use_gpu=True, force_gpu=True):
      tf.global_variables_initializer().run()
      self.assertAllClose(o1.eval(), o2.eval(), atol=atol)

  @parameterized.named_parameters(
      ("WithBias_Stride1", True, 1), ("WithoutBias_Stride1", False, 1),
      ("WithBias_Stride2", True, 2), ("WithoutBias_Stride2", False, 2))
  def testConv2DTransposeDataFormats(self, use_bias, stride):
    """Check the module produces the same result for supported data formats."""
    input_shape = (self.INPUT_SHAPE.input_batch,
                   int(np.ceil(self.INPUT_SHAPE.input_height / stride)),
                   int(np.ceil(self.INPUT_SHAPE.input_width / stride)),
                   self.INPUT_SHAPE.input_channels)

    func = functools.partial(
        snt.Conv2DTranspose,
        output_channels=self.OUT_CHANNELS,
        kernel_shape=self.KERNEL_SHAPE,
        output_shape=(self.INPUT_SHAPE.input_height,
                      self.INPUT_SHAPE.input_width),
        use_bias=use_bias,
        stride=stride,
        initializers=create_initializers(use_bias))

    conv_nhwc = func(name="NHWC", data_format="NHWC")
    x = tf.constant(np.random.random(input_shape).astype(np.float32))
    result_nhwc = conv_nhwc(x)

    # We will force both modules to share the same weights by creating
    # a custom getter that returns the weights from the first conv module when
    # tf.get_variable is called.
    custom_getter = {"w": create_custom_field_getter(conv_nhwc, "w"),
                     "b": create_custom_field_getter(conv_nhwc, "b")}
    conv_nchw = func(name="NCHW", data_format="NCHW",
                     custom_getter=custom_getter)
    x_transpose = tf.transpose(x, perm=(0, 3, 1, 2))
    result_nchw = tf.transpose(conv_nchw(x_transpose), perm=(0, 2, 3, 1))

    self.checkEquality(result_nhwc, result_nchw)

  @parameterized.named_parameters(("WithBias", True), ("WithoutBias", False))
  def testConv2DTransposeDataFormatsBatchNorm(self, use_bias):
    """Like `testConv2DTransposeDataFormats` but checks BatchNorm support."""

    def func(name, data_format, custom_getter=None):
      conv = snt.Conv2DTranspose(
          name=name,
          output_channels=self.OUT_CHANNELS,
          kernel_shape=self.KERNEL_SHAPE,
          output_shape=(self.INPUT_SHAPE.input_height,
                        self.INPUT_SHAPE.input_width),
          use_bias=use_bias,
          initializers=create_initializers(use_bias),
          data_format=data_format,
          custom_getter=custom_getter)
      if data_format == "NHWC":
        batch_norm = snt.BatchNorm(scale=True, update_ops_collection=None)
      else:  # data_format == "NCHW"
        batch_norm = snt.BatchNorm(scale=True, update_ops_collection=None,
                                   fused=True, axis=(0, 2, 3))
      return snt.Sequential([conv,
                             functools.partial(batch_norm, is_training=True)])

    seq_nhwc = func(name="NHWC", data_format="NHWC")
    x = tf.constant(np.random.random(self.INPUT_SHAPE).astype(np.float32))
    result_nhwc = seq_nhwc(x)

    custom_getter = {"w": create_custom_field_getter(seq_nhwc.layers[0], "w"),
                     "b": create_custom_field_getter(seq_nhwc.layers[0], "b")}
    seq_nchw = func(name="NCHW", data_format="NCHW",
                    custom_getter=custom_getter)
    x_transpose = tf.transpose(x, perm=(0, 3, 1, 2))
    result_nchw = tf.transpose(seq_nchw(x_transpose), perm=(0, 2, 3, 1))

    self.checkEquality(result_nhwc, result_nchw)


class Conv3DTransposeTestDataFormats(parameterized.TestCase, tf.test.TestCase):
  OUT_CHANNELS = 5
  KERNEL_SHAPE = 3
  INPUT_SHAPE = Conv3DInput(2, 17, 18, 19, 4)

  def setUp(self):
    super(Conv3DTransposeTestDataFormats, self).setUp()
    name = "{}.{}".format(type(self).__name__, self._testMethodName)
    if not test.is_gpu_available():
      self.skipTest("No GPU was detected, so {} will be skipped.".format(name))

  def checkEquality(self, o1, o2, atol=1e-5):
    with self.test_session(use_gpu=True, force_gpu=True):
      tf.global_variables_initializer().run()
      self.assertAllClose(o1.eval(), o2.eval(), atol=atol)

  @parameterized.named_parameters(
      ("WithBias_Stride1", True, 1), ("WithoutBias_Stride1", False, 1),
      ("WithBias_Stride2", True, 2), ("WithoutBias_Stride2", False, 2))
  def testConv3DTransposeDataFormats(self, use_bias, stride):
    """Check the module produces the same result for supported data formats."""
    input_shape = (self.INPUT_SHAPE.input_batch,
                   int(np.ceil(self.INPUT_SHAPE.input_depth / stride)),
                   int(np.ceil(self.INPUT_SHAPE.input_height / stride)),
                   int(np.ceil(self.INPUT_SHAPE.input_width / stride)),
                   self.INPUT_SHAPE.input_channels)

    func = functools.partial(
        snt.Conv3DTranspose,
        output_channels=self.OUT_CHANNELS,
        kernel_shape=self.KERNEL_SHAPE,
        output_shape=(self.INPUT_SHAPE.input_depth,
                      self.INPUT_SHAPE.input_height,
                      self.INPUT_SHAPE.input_width),
        use_bias=use_bias,
        stride=stride,
        initializers=create_initializers(use_bias))

    conv_ndhwc = func(name="NDHWC", data_format="NDHWC")
    x = tf.constant(np.random.random(input_shape).astype(np.float32))
    result_ndhwc = conv_ndhwc(x)

    # We will force both modules to share the same weights by creating
    # a custom getter that returns the weights from the first conv module when
    # tf.get_variable is called.
    custom_getter = {"w": create_custom_field_getter(conv_ndhwc, "w"),
                     "b": create_custom_field_getter(conv_ndhwc, "b")}
    conv_ncdhw = func(name="NCDHW", data_format="NCDHW",
                      custom_getter=custom_getter)
    x_transpose = tf.transpose(x, perm=(0, 4, 1, 2, 3))
    result_ncdhw = tf.transpose(conv_ncdhw(x_transpose), perm=(0, 2, 3, 4, 1))

    self.checkEquality(result_ndhwc, result_ncdhw)

  @parameterized.named_parameters(("WithBias", True), ("WithoutBias", False))
  def testConv3DTransposeDataFormatsBatchNorm(self, use_bias):
    """Like `testConv3DTransposeDataFormats` but checks BatchNorm support."""

    def func(name, data_format, custom_getter=None):
      conv = snt.Conv3DTranspose(
          name=name,
          output_channels=self.OUT_CHANNELS,
          kernel_shape=self.KERNEL_SHAPE,
          output_shape=(self.INPUT_SHAPE.input_depth,
                        self.INPUT_SHAPE.input_height,
                        self.INPUT_SHAPE.input_width),
          use_bias=use_bias,
          initializers=create_initializers(use_bias),
          data_format=data_format,
          custom_getter=custom_getter)
      if data_format == "NDHWC":
        batch_norm = snt.BatchNorm(scale=True, update_ops_collection=None)
      else:  # data_format == "NCDHW"
        batch_norm = snt.BatchNorm(scale=True, update_ops_collection=None,
                                   axis=(0, 2, 3, 4))
      return snt.Sequential([conv,
                             functools.partial(batch_norm, is_training=True)])

    seq_ndhwc = func(name="NDHWC", data_format="NDHWC")
    x = tf.constant(np.random.random(self.INPUT_SHAPE).astype(np.float32))
    result_ndhwc = seq_ndhwc(x)

    custom_getter = {"w": create_custom_field_getter(seq_ndhwc.layers[0], "w"),
                     "b": create_custom_field_getter(seq_ndhwc.layers[0], "b")}
    seq_ncdhw = func(name="NCDHW", data_format="NCDHW",
                     custom_getter=custom_getter)
    x_transpose = tf.transpose(x, perm=(0, 4, 1, 2, 3))
    result_ncdhw = tf.transpose(seq_ncdhw(x_transpose), perm=(0, 2, 3, 4, 1))

    self.checkEquality(result_ndhwc, result_ncdhw)

if __name__ == "__main__":
  tf.test.main()
