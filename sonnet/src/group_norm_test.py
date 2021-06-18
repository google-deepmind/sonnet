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
"""Tests for sonnet.v2.src.group_norm."""

from absl.testing import parameterized
import numpy as np
from sonnet.src import group_norm
from sonnet.src import initializers
from sonnet.src import test_utils
import tensorflow as tf


class GroupNormTest(test_utils.TestCase, parameterized.TestCase):

  def testSimpleCase(self):
    layer = group_norm.GroupNorm(
        groups=5, create_scale=False, create_offset=False)
    inputs = tf.ones([2, 3, 3, 10])

    outputs = layer(inputs).numpy()
    for x in np.nditer(outputs):
      self.assertEqual(x, 0.0)

  def testSimpleCaseVar(self):
    layer = group_norm.GroupNorm(
        groups=5,
        create_scale=True,
        create_offset=True,
        scale_init=initializers.Constant(0.5),
        offset_init=initializers.Constant(2.0))

    inputs = tf.ones([2, 3, 3, 10])

    outputs = layer(inputs).numpy()
    for x in np.nditer(outputs):
      self.assertEqual(x, 2.0)

  def testSimpleCaseNCHWVar(self):
    layer = group_norm.GroupNorm(
        groups=5,
        create_scale=True,
        create_offset=True,
        scale_init=initializers.Constant(0.5),
        offset_init=initializers.Constant(2.0),
        data_format="NCHW")

    inputs = tf.ones([2, 10, 3, 3])

    outputs = layer(inputs).numpy()
    for x in np.nditer(outputs):
      self.assertEqual(x, 2.0)

  def testDataFormatAgnosticVar(self):
    c_last_layer = group_norm.GroupNorm(
        groups=5, create_scale=True, create_offset=True)
    c_first_layer = group_norm.GroupNorm(
        groups=5, create_scale=True, create_offset=True, data_format="NCHW")

    inputs = tf.random.uniform([3, 4, 4, 10], 0, 10)

    c_last_output = c_last_layer(inputs)
    inputs = tf.transpose(inputs, [0, 3, 1, 2])
    c_first_output = c_first_layer(inputs)
    c_first_output = tf.transpose(c_first_output, [0, 2, 3, 1])

    self.assertAllClose(c_last_output.numpy(), c_first_output.numpy())

  def testSimpleCaseTensor(self):
    layer = group_norm.GroupNorm(
        groups=5, create_scale=False, create_offset=False)

    inputs = tf.ones([2, 3, 3, 10])
    scale = tf.constant(0.5, shape=(10,))
    offset = tf.constant(2.0, shape=(10,))

    outputs = layer(inputs, scale, offset).numpy()
    for x in np.nditer(outputs):
      self.assertEqual(x, 2.0)

  def testSimpleCaseNCHWTensor(self):
    layer = group_norm.GroupNorm(
        groups=5, data_format="NCHW", create_scale=False, create_offset=False)

    inputs = tf.ones([2, 10, 3, 3])
    scale = tf.constant(0.5, shape=(10, 1, 1))
    offset = tf.constant(2.0, shape=(10, 1, 1))

    outputs = layer(inputs, scale, offset).numpy()
    for x in np.nditer(outputs):
      self.assertEqual(x, 2.0)

  def testDataFormatAgnosticTensor(self):
    c_last = group_norm.GroupNorm(
        groups=5, create_scale=False, create_offset=False)
    c_first = group_norm.GroupNorm(
        groups=5, data_format="NCHW", create_scale=False, create_offset=False)

    inputs = tf.random.uniform([3, 4, 4, 10], 0, 10)
    scale = tf.random.normal((10,), mean=1.0)
    offset = tf.random.normal((10,))

    c_last_output = c_last(inputs, scale, offset)
    inputs = tf.transpose(inputs, [0, 3, 1, 2])
    scale = tf.reshape(scale, (10, 1, 1))
    offset = tf.reshape(offset, (10, 1, 1))
    c_first_output = c_first(inputs, scale, offset)
    c_first_output = tf.transpose(c_first_output, [0, 2, 3, 1])

    self.assertAllClose(c_last_output, c_first_output, rtol=1e-5)

  @parameterized.parameters("NHW", "HWC", "channel_last")
  def testInvalidDataFormat(self, data_format):
    with self.assertRaisesRegex(
        ValueError,
        "Unable to extract channel information from '{}'.".format(data_format)):
      group_norm.GroupNorm(
          groups=5,
          data_format=data_format,
          create_scale=False,
          create_offset=False)

  @parameterized.parameters("NCHW", "NCW", "channels_first")
  def testValidDataFormatChannelsFirst(self, data_format):
    test = group_norm.GroupNorm(
        groups=5,
        data_format=data_format,
        create_scale=False,
        create_offset=False)

    self.assertEqual(test._channel_index, 1)

  @parameterized.parameters("NHWC", "NWC", "channels_last")
  def testValidDataFormatChannelsLast(self, data_format):
    test = group_norm.GroupNorm(
        groups=5,
        data_format=data_format,
        create_scale=False,
        create_offset=False)

    self.assertEqual(test._channel_index, -1)

  @parameterized.named_parameters(("String", "foo"), ("ListString", ["foo"]))
  def testInvalidAxis(self, axis):
    with self.assertRaisesRegex(
        ValueError, "`axis` should be an int, slice or iterable of ints."):
      group_norm.GroupNorm(
          groups=5, axis=axis, create_scale=False, create_offset=False)

  def testNoScaleAndInitProvided(self):
    with self.assertRaisesRegex(
        ValueError, "Cannot set `scale_init` if `create_scale=False`."):
      group_norm.GroupNorm(
          groups=5,
          create_scale=False,
          create_offset=True,
          scale_init=initializers.Ones())

  def testNoOffsetBetaInitProvided(self):
    with self.assertRaisesRegex(
        ValueError, "Cannot set `offset_init` if `create_offset=False`."):
      group_norm.GroupNorm(
          groups=5,
          create_scale=True,
          create_offset=False,
          offset_init=initializers.Zeros())

  def testCreateScaleAndScaleProvided(self):
    layer = group_norm.GroupNorm(
        groups=5, create_scale=True, create_offset=False)

    with self.assertRaisesRegex(
        ValueError, "Cannot pass `scale` at call time if `create_scale=True`."):
      layer(tf.ones([2, 3, 5]), scale=tf.ones([4]))

  def testCreateOffsetAndOffsetProvided(self):
    layer = group_norm.GroupNorm(
        groups=5, create_offset=True, create_scale=False)

    with self.assertRaisesRegex(
        ValueError,
        "Cannot pass `offset` at call time if `create_offset=True`."):
      layer(tf.ones([2, 3, 5]), offset=tf.ones([4]))

  def testSliceAxis(self):
    slice_layer = group_norm.GroupNorm(
        groups=5, create_scale=False, create_offset=False)
    axis_layer = group_norm.GroupNorm(
        groups=5, create_scale=False, create_offset=False)

    inputs = tf.random.uniform([3, 4, 4, 5], 0, 10)
    scale = tf.random.normal((5,), mean=1.0)
    offset = tf.random.normal((5,))

    slice_outputs = slice_layer(inputs, scale, offset)
    axis_outputs = axis_layer(inputs, scale, offset)

    self.assertAllEqual(slice_outputs.numpy(), axis_outputs.numpy())

  def testRankChanges(self):
    layer = group_norm.GroupNorm(
        groups=5, create_scale=False, create_offset=False)

    inputs = tf.ones([2, 3, 3, 5])
    scale = tf.constant(0.5, shape=(5,))
    offset = tf.constant(2.0, shape=(5,))

    layer(inputs, scale, offset)

    with self.assertRaisesRegex(
        ValueError,
        "The rank of the inputs cannot change between calls, the original"):
      layer(tf.ones([2, 3, 3, 4, 5]), scale, offset)

  @parameterized.named_parameters(("Small", (2, 4, 4)), ("Bigger", (2, 3, 8)))
  def testIncompatibleGroupsAndTensor(self, shape):
    layer = group_norm.GroupNorm(
        groups=5, create_scale=False, create_offset=False)

    inputs = tf.ones(shape)

    with self.assertRaisesRegex(
        ValueError,
        "The number of channels must be divisible by the number of groups"):
      layer(inputs)

  def testWorksWithFunction(self):
    layer = group_norm.GroupNorm(
        groups=5, create_scale=False, create_offset=False)
    function_layer = tf.function(layer)

    inputs = tf.ones([2, 3, 3, 10])
    scale = tf.constant(0.5, shape=(10,))
    offset = tf.constant(2.0, shape=(10,))

    outputs = layer(inputs, scale, offset)
    function_outputs = function_layer(inputs, scale, offset)

    self.assertAllEqual(outputs.numpy(), function_outputs.numpy())

  def testBatchSizeAgnostic(self):
    layer = group_norm.GroupNorm(
        groups=5, create_scale=False, create_offset=False)
    inputs_spec = tf.TensorSpec([None, 3, 3, 10], dtype=tf.float32)
    params_spec = tf.TensorSpec([None], dtype=tf.float32)
    function_layer = tf.function(layer).get_concrete_function(
        inputs_spec, params_spec, params_spec)

    scale = tf.constant(0.5, shape=(10,))
    offset = tf.constant(2.0, shape=(10,))

    outputs = function_layer(tf.ones([2, 3, 3, 10]), scale, offset)
    self.assertEqual(outputs.shape, [2, 3, 3, 10])
    for x in np.nditer(outputs):
      self.assertEqual(x, 2.0)

    scale = tf.constant(0.5, shape=(10,))
    offset = tf.constant(2.0, shape=(10,))

    outputs = function_layer(tf.ones([3, 3, 3, 10]), scale, offset)
    self.assertEqual(outputs.shape, [3, 3, 3, 10])
    for x in np.nditer(outputs):
      self.assertEqual(x, 2.0)

  def test5DDataFormatAgnostic(self):
    c_last_layer = group_norm.GroupNorm(
        groups=5, create_scale=False, create_offset=False)
    c_first_layer = group_norm.GroupNorm(
        groups=5, create_scale=False, create_offset=False, data_format="NCDHW")

    inputs = tf.random.uniform([3, 4, 4, 4, 10], 0, 10)
    scale = tf.random.normal((10,), mean=1.0)
    offset = tf.random.normal((10,))

    c_last_output = c_last_layer(inputs, scale, offset)
    inputs = tf.transpose(inputs, [0, 4, 1, 2, 3])
    scale = tf.reshape(scale, [-1, 1, 1, 1])
    offset = tf.reshape(offset, [-1, 1, 1, 1])
    c_first_output = c_first_layer(inputs, scale, offset)
    c_first_output = tf.transpose(c_first_output, [0, 2, 3, 4, 1])

    self.assertAllClose(
        c_last_output.numpy(), c_first_output.numpy(), atol=1e-5, rtol=1e-5)

  def test3DDataFormatAgnostic(self):
    c_last_layer = group_norm.GroupNorm(
        groups=5, create_scale=False, create_offset=False)
    c_first_layer = group_norm.GroupNorm(
        groups=5, create_scale=False, create_offset=False, data_format="NCW")

    inputs = tf.random.uniform([3, 4, 10], 0, 10)
    scale = tf.random.normal((10,), mean=1.0)
    offset = tf.random.normal((10,))

    c_last_output = c_last_layer(inputs, scale, offset)
    inputs = tf.transpose(inputs, [0, 2, 1])
    scale = tf.reshape(scale, [-1, 1])
    offset = tf.reshape(offset, [-1, 1])
    c_first_output = c_first_layer(inputs, scale, offset)
    c_first_output = tf.transpose(c_first_output, [0, 2, 1])

    self.assertAllClose(
        c_last_output.numpy(), c_first_output.numpy(), atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
  tf.test.main()
