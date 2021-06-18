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
"""Tests for sonnet.v2.src.axis_norm."""

from absl.testing import parameterized
import numpy as np
from sonnet.src import axis_norm
from sonnet.src import initializers
from sonnet.src import test_utils
import tensorflow as tf


class LayerNormTest(test_utils.TestCase, parameterized.TestCase):

  def testSimpleCase(self):
    layer = axis_norm.LayerNorm([1, 2], create_scale=False, create_offset=False)
    inputs = tf.ones([2, 3, 3, 5])

    outputs = layer(inputs).numpy()
    for x in np.nditer(outputs):
      self.assertEqual(x, 0.0)

  def testSimpleCaseVar(self):
    layer = axis_norm.LayerNorm([1, 2],
                                create_scale=True,
                                create_offset=True,
                                scale_init=initializers.Constant(0.5),
                                offset_init=initializers.Constant(2.0))

    inputs = tf.ones([2, 3, 3, 5])

    outputs = layer(inputs).numpy()
    for x in np.nditer(outputs):
      self.assertEqual(x, 2.0)

  def testSimpleCaseNCHWVar(self):
    layer = axis_norm.LayerNorm([1, 2],
                                create_scale=True,
                                create_offset=True,
                                scale_init=initializers.Constant(0.5),
                                offset_init=initializers.Constant(2.0),
                                data_format="NCHW")

    inputs = tf.ones([2, 5, 3, 3])

    outputs = layer(inputs).numpy()
    for x in np.nditer(outputs):
      self.assertEqual(x, 2.0)

  def testDataFormatAgnosticVar(self):
    c_last_layer = axis_norm.LayerNorm([1, 2],
                                       create_scale=True,
                                       create_offset=True)
    c_first_layer = axis_norm.LayerNorm([2, 3],
                                        create_scale=True,
                                        create_offset=True,
                                        data_format="NCHW")

    inputs = tf.random.uniform([3, 4, 4, 5], 0, 10)

    c_last_output = c_last_layer(inputs)
    inputs = tf.transpose(inputs, [0, 3, 1, 2])
    c_first_output = c_first_layer(inputs)
    c_first_output = tf.transpose(c_first_output, [0, 2, 3, 1])

    self.assertAllClose(c_last_output.numpy(), c_first_output.numpy())

  def testSimpleCaseTensor(self):
    layer = axis_norm.LayerNorm([1, 2], create_scale=False, create_offset=False)

    inputs = tf.ones([2, 3, 3, 5])
    scale = tf.constant(0.5, shape=(5,))
    offset = tf.constant(2.0, shape=(5,))

    outputs = layer(inputs, scale, offset).numpy()
    for x in np.nditer(outputs):
      self.assertEqual(x, 2.0)

  def testSimpleCaseNCHWTensor(self):
    layer = axis_norm.LayerNorm([1, 2],
                                data_format="NCHW",
                                create_scale=False,
                                create_offset=False)

    inputs = tf.ones([2, 5, 3, 3])
    scale = tf.constant(0.5, shape=(5, 1, 1))
    offset = tf.constant(2.0, shape=(5, 1, 1))

    outputs = layer(inputs, scale, offset).numpy()
    for x in np.nditer(outputs):
      self.assertEqual(x, 2.0)

  def testDataFormatAgnosticTensor(self):
    c_last_layer = axis_norm.LayerNorm([1, 2],
                                       create_scale=False,
                                       create_offset=False)
    c_first_layer = axis_norm.LayerNorm([2, 3],
                                        data_format="NCHW",
                                        create_scale=False,
                                        create_offset=False)

    inputs = tf.random.uniform([3, 4, 4, 5], 0, 10)
    scale = tf.random.normal((5,), mean=1.0)
    offset = tf.random.normal((5,))

    c_last_output = c_last_layer(inputs, scale, offset)
    inputs = tf.transpose(inputs, [0, 3, 1, 2])
    scale = tf.reshape(scale, (5, 1, 1))
    offset = tf.reshape(offset, (5, 1, 1))
    c_first_output = c_first_layer(inputs, scale, offset)
    c_first_output = tf.transpose(c_first_output, [0, 2, 3, 1])

    self.assertAllClose(c_last_output.numpy(), c_first_output.numpy())

  @parameterized.parameters("NHW", "HWC", "channel_last")
  def testInvalidDataFormat(self, data_format):
    with self.assertRaisesRegex(
        ValueError,
        "Unable to extract channel information from '{}'.".format(data_format)):
      axis_norm.LayerNorm(
          3, data_format=data_format, create_scale=False, create_offset=False)

  @parameterized.parameters("NCHW", "NCW", "channels_first")
  def testValidDataFormatChannelsFirst(self, data_format):
    test = axis_norm.LayerNorm(
        3, data_format=data_format, create_scale=False, create_offset=False)

    self.assertEqual(test._channel_index, 1)

  @parameterized.parameters("NHWC", "NWC", "channels_last")
  def testValidDataFormatChannelsLast(self, data_format):
    test = axis_norm.LayerNorm(
        3, data_format=data_format, create_scale=False, create_offset=False)

    self.assertEqual(test._channel_index, -1)

  @parameterized.named_parameters(("String", "foo"), ("ListString", ["foo"]))
  def testInvalidAxis(self, axis):
    with self.assertRaisesRegex(
        ValueError, "`axis` should be an int, slice or iterable of ints."):
      axis_norm.LayerNorm(axis, create_scale=False, create_offset=False)

  def testNoScaleAndInitProvided(self):
    with self.assertRaisesRegex(
        ValueError, "Cannot set `scale_init` if `create_scale=False`."):
      axis_norm.LayerNorm(
          3,
          create_scale=False,
          create_offset=True,
          scale_init=initializers.Ones())

  def testNoOffsetBetaInitProvided(self):
    with self.assertRaisesRegex(
        ValueError, "Cannot set `offset_init` if `create_offset=False`."):
      axis_norm.LayerNorm(
          3,
          create_scale=True,
          create_offset=False,
          offset_init=initializers.Zeros())

  def testCreateScaleAndScaleProvided(self):
    layer = axis_norm.LayerNorm([2], create_scale=True, create_offset=False)

    with self.assertRaisesRegex(
        ValueError, "Cannot pass `scale` at call time if `create_scale=True`."):
      layer(tf.ones([2, 3, 4]), scale=tf.ones([4]))

  def testCreateOffsetAndOffsetProvided(self):
    layer = axis_norm.LayerNorm([2], create_offset=True, create_scale=False)

    with self.assertRaisesRegex(
        ValueError,
        "Cannot pass `offset` at call time if `create_offset=True`."):
      layer(tf.ones([2, 3, 4]), offset=tf.ones([4]))

  def testSliceAxis(self):
    slice_layer = axis_norm.LayerNorm(
        slice(1, -1), create_scale=False, create_offset=False)
    axis_layer = axis_norm.LayerNorm((1, 2),
                                     create_scale=False,
                                     create_offset=False)

    inputs = tf.random.uniform([3, 4, 4, 5], 0, 10)
    scale = tf.random.normal((5,), mean=1.0)
    offset = tf.random.normal((5,))

    slice_outputs = slice_layer(inputs, scale, offset)
    axis_outputs = axis_layer(inputs, scale, offset)

    self.assertAllEqual(slice_outputs.numpy(), axis_outputs.numpy())

  def testRankChanges(self):
    layer = axis_norm.LayerNorm((1, 2), create_scale=False, create_offset=False)

    inputs = tf.ones([2, 3, 3, 5])
    scale = tf.constant(0.5, shape=(5,))
    offset = tf.constant(2.0, shape=(5,))

    layer(inputs, scale, offset)

    with self.assertRaisesRegex(
        ValueError,
        "The rank of the inputs cannot change between calls, the original"):
      layer(tf.ones([2, 3, 3, 4, 5]), scale, offset)

  def testWorksWithFunction(self):
    layer = axis_norm.LayerNorm((1, 2), create_scale=False, create_offset=False)
    function_layer = tf.function(layer)

    inputs = tf.ones([2, 3, 3, 5])
    scale = tf.constant(0.5, shape=(5,))
    offset = tf.constant(2.0, shape=(5,))

    outputs = layer(inputs, scale, offset)
    function_outputs = function_layer(inputs, scale, offset)

    self.assertAllEqual(outputs.numpy(), function_outputs.numpy())

  def testShapeAgnostic(self):
    layer = axis_norm.LayerNorm((1, 2), create_scale=False, create_offset=False)
    inputs_spec = tf.TensorSpec([None, None, None, None], dtype=tf.float32)
    params_spec = tf.TensorSpec([None], dtype=tf.float32)
    function_layer = tf.function(layer).get_concrete_function(
        inputs_spec, params_spec, params_spec)

    scale = tf.constant(0.5, shape=(5,))
    offset = tf.constant(2.0, shape=(5,))

    outputs = function_layer(tf.ones([2, 3, 3, 5]), scale, offset)
    self.assertEqual(outputs.shape, [2, 3, 3, 5])
    for x in np.nditer(outputs):
      self.assertEqual(x, 2.0)

    scale = tf.constant(0.5, shape=(3,))
    offset = tf.constant(2.0, shape=(3,))

    outputs = function_layer(tf.ones([3, 4, 6, 3]), scale, offset)
    self.assertEqual(outputs.shape, [3, 4, 6, 3])
    for x in np.nditer(outputs):
      self.assertEqual(x, 2.0)

  def test5DDataFormatAgnostic(self):
    c_last_layer = axis_norm.LayerNorm([1, 2, 3],
                                       create_scale=False,
                                       create_offset=False)
    c_first_layer = axis_norm.LayerNorm([2, 3, 4],
                                        create_scale=False,
                                        create_offset=False,
                                        data_format="NCDHW")

    inputs = tf.random.uniform([3, 4, 4, 4, 5], 0, 10)
    scale = tf.random.normal((5,), mean=1.0)
    offset = tf.random.normal((5,))

    c_last_output = c_last_layer(inputs, scale, offset)
    inputs = tf.transpose(inputs, [0, 4, 1, 2, 3])
    scale = tf.reshape(scale, [-1, 1, 1, 1])
    offset = tf.reshape(offset, [-1, 1, 1, 1])
    c_first_output = c_first_layer(inputs, scale, offset)
    c_first_output = tf.transpose(c_first_output, [0, 2, 3, 4, 1])

    self.assertAllClose(
        c_last_output.numpy(), c_first_output.numpy(), atol=1e-5, rtol=1e-5)

  def test3DDataFormatAgnostic(self):
    c_last_layer = axis_norm.LayerNorm([1],
                                       create_scale=False,
                                       create_offset=False)
    c_first_layer = axis_norm.LayerNorm([2],
                                        create_scale=False,
                                        create_offset=False,
                                        data_format="NCW")

    inputs = tf.random.uniform([3, 4, 5], 0, 10)
    scale = tf.random.normal((5,), mean=1.0)
    offset = tf.random.normal((5,))

    c_last_output = c_last_layer(inputs, scale, offset)
    inputs = tf.transpose(inputs, [0, 2, 1])
    scale = tf.reshape(scale, [-1, 1])
    offset = tf.reshape(offset, [-1, 1])
    c_first_output = c_first_layer(inputs, scale, offset)
    c_first_output = tf.transpose(c_first_output, [0, 2, 1])

    self.assertAllClose(
        c_last_output.numpy(), c_first_output.numpy(), atol=1e-5, rtol=1e-5)

  def testInstanceNormCorrectAxis(self):
    layer = axis_norm.InstanceNorm(create_scale=True, create_offset=True)

    inputs = tf.ones([3, 4, 5, 6])
    layer(inputs)

    self.assertEqual(layer._axis, (1, 2))

  def testInstanceNormCorrectNCW(self):
    layer = axis_norm.InstanceNorm(
        create_scale=True, create_offset=True, data_format="channels_first")

    inputs = tf.ones([3, 4, 5, 6])
    layer(inputs)

    self.assertEqual(layer._axis, (2, 3))


if __name__ == "__main__":
  tf.test.main()
