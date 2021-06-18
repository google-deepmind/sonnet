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
"""Tests for sonnet.v2.src.reshape."""

from absl.testing import parameterized
import numpy as np
from sonnet.src import reshape
from sonnet.src import test_utils
import tensorflow as tf

B, H, W, C, D = 2, 3, 4, 5, 6


class ReshapeTest(test_utils.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      (1, [B, H * W * C, D]),
      (2, [B, H, W * C, D]),
      (3, [B, H, W, C, D]),
      (4, [B, H, W, C, 1, D]),
  )
  def testReshape(self, preserve_dims, expected_output_shape):
    mod = reshape.Reshape(output_shape=(-1, D), preserve_dims=preserve_dims)
    outputs = mod(tf.ones([B, H, W, C, D]))
    self.assertEqual(outputs.shape, expected_output_shape)

  def testInvalid_multipleWildcard(self):
    mod = reshape.Reshape(output_shape=[-1, -1])
    with self.assertRaises(tf.errors.InvalidArgumentError):
      mod(tf.ones([1, 2, 3]))

  def testInvalid_negativeSize(self):
    mod = reshape.Reshape(output_shape=[1, -2])
    with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                "[Ss]ize 2 must be non-negative, not -2"):
      mod(tf.ones([1, 2, 3]))

  def testInvalid_type(self):
    mod = reshape.Reshape(output_shape=[7, "string"])
    with self.assertRaises(ValueError):
      mod(tf.ones([1, 2, 3]))

  def testIncompatibleShape(self):
    mod = reshape.Reshape(output_shape=[2 * 3, 4])

    input_size = 8 * 2 * 2 * 4
    output_size = 8 * 2 * 3 * 4
    msg = ("Input to reshape is a tensor with %d values, "
           "but the requested shape has %d" % (input_size, output_size))
    with self.assertRaisesRegex(tf.errors.InvalidArgumentError, msg):
      mod(tf.ones([8, 2, 2, 4]))

  def testInferShape(self):
    batch_size = 10
    out_size = [2, -1, 5]
    mod = reshape.Reshape(output_shape=out_size)
    output = mod(tf.ones([batch_size, 2, 3, 4, 5]))
    self.assertEqual(output.shape, [batch_size, 2, 3 * 4, 5])

  def testAddDimensions(self):
    batch_size = 10

    mod = reshape.Reshape(output_shape=[1, 1])
    inputs = tf.ones([batch_size])
    output = mod(inputs)
    self.assertEqual(output.shape, [batch_size, 1, 1])

    # Reverse should remove the additional dims.
    mod_t = mod.reversed()
    t_output = mod_t(output)
    self.assertEqual(t_output.shape, [batch_size])

  def testFlatten(self):
    batch_size = 10
    inputs = tf.ones([batch_size, 2, 3, 4, 5])
    mod = reshape.Reshape(output_shape=[-1])
    output = mod(inputs)
    self.assertEqual(output.shape, [batch_size, 2 * 3 * 4 * 5])

  def testUnknownBatchSize(self):
    mod = reshape.Reshape(output_shape=[-1])
    input_spec = tf.TensorSpec([None, 2, 3, 4, 5], tf.float32)
    cf = tf.function(mod).get_concrete_function(input_spec)
    output, = cf.outputs
    self.assertEqual(output.shape.as_list(), [None, 2 * 3 * 4 * 5])

  def testReverse(self):
    batch_size = 10
    input_shape = [batch_size, 2, 3, 4, 5]
    expected_output_shape = [batch_size, 2, 3 * 4, 5]

    inputs = tf.random.normal(input_shape)
    mod = reshape.Reshape(output_shape=[2, -1, 5])
    output = mod(inputs)
    self.assertEqual(output.shape, expected_output_shape)

    mod_r = mod.reversed()
    output_r = mod_r(output)
    self.assertEqual(output_r.shape, input_shape)

    mod_r_r = mod_r.reversed()
    output_r_r = mod_r_r(output)
    self.assertEqual(output_r_r.shape, expected_output_shape)

    input_np, output_r_np = self.evaluate([inputs, output_r])
    self.assertAllClose(output_r_np, input_np)

  def testReverse_name(self):
    mod = reshape.Reshape(output_shape=[2, -1, 5])
    mod(tf.ones([1, 2, 3, 4, 5]))
    mod_r = mod.reversed()
    self.assertEqual(mod_r.name, "%s_reversed" % mod.name)

  def testInvalidPreserveDimsError(self):
    with self.assertRaisesRegex(ValueError, "preserve_dims"):
      reshape.Reshape((-1,), preserve_dims=0)

  def testBuildDimError(self):
    mod = reshape.Reshape((-1,), preserve_dims=2)
    input_tensor = tf.ones([50])
    with self.assertRaisesRegex(ValueError, "preserve_dims"):
      mod(input_tensor)

  @parameterized.named_parameters(
      ("Preserve1", (1,)),
      ("Preserve24", (2, 4)),
      ("Preserve?", (None,)),
      ("Preserve?5", (None, 5)),
      ("Preserve5?", (5, None)),
      ("Preserve??", (None, None)),
  )
  def testPreserve(self, preserve):
    shape = list(preserve) + [13, 84, 3, 2]
    output_shape = [13, 21, 3, 8]
    preserve_dims = len(preserve)
    input_spec = tf.TensorSpec(shape, tf.float32)
    mod = reshape.Reshape(
        output_shape=output_shape, preserve_dims=preserve_dims)
    cf = tf.function(mod).get_concrete_function(input_spec)
    output, = cf.outputs
    self.assertEqual(output.shape.as_list(), list(preserve) + output_shape)

  @parameterized.named_parameters(
      ("Session1", (1,), (2, 3), (-1,)),
      ("Session2", (1, 7), (2, 3), (-1,)),
      ("Session3", (None,), (2, 3), (-1,)),
      ("Session4", (None, 5, None), (2, 3, 4), (4, 6)),
      ("Session5", (None, None, None), (2, 3, 4), (-1,)),
      ("Session6", (5, None, None), (1, 3, 1), (-1,)),
      ("Session7", (1,), (4, 3), (2, 2, 1, 3)),
      ("Session8", (None,), (4, 3), (2, 2, 1, 3)),
      ("Session9", (1, None, 5, None), (4, 3), (2, 2, -1, 3)),
  )
  def testRun(self, preserve, trailing_in, trailing_out):
    rng = np.random.RandomState(0)
    input_shape = preserve + trailing_in
    output_shape = preserve + np.zeros(trailing_in).reshape(trailing_out).shape
    input_spec = tf.TensorSpec(input_shape, tf.float32)
    mod = reshape.Reshape(
        output_shape=trailing_out, preserve_dims=len(preserve))
    cf = tf.function(mod).get_concrete_function(input_spec)
    output, = cf.outputs
    self.assertEqual(output.shape.as_list(), list(output_shape))

    actual_input_shape = [13 if i is None else i for i in input_shape]
    expected_output_shape = [13 if i is None else i for i in output_shape]
    actual_input = rng.rand(*actual_input_shape).astype(np.float32)
    expected_output = actual_input.reshape(expected_output_shape)
    actual_output = cf(tf.convert_to_tensor(actual_input))
    self.assertAllEqual(actual_output, expected_output)


class FlattenTest(test_utils.TestCase, parameterized.TestCase):

  @parameterized.parameters([1, 10])
  def testFlatten(self, batch_size):
    in_shape = [2, 3, 4, 5]
    inputs = tf.ones([batch_size] + in_shape)
    mod = reshape.Flatten()
    output = mod(inputs)
    flattened_size = np.prod(in_shape, dtype=int)
    self.assertEqual(output.shape, [batch_size, flattened_size])

  def testFlatten_unknownBatchSize(self):
    mod = reshape.Flatten()
    f = tf.function(mod)
    inputs = tf.TensorSpec([None, 1, 2, 3], tf.float32)
    cf = f.get_concrete_function(inputs)
    self.assertEqual(cf.outputs[0].shape.as_list(), [None, 1 * 2 * 3])
    flat = cf(tf.ones([8, 1, 2, 3]))
    self.assertEqual(flat.shape, [8, 1 * 2 * 3])

  def testFlatten_unknownNonBatchSize(self):
    mod = reshape.Flatten()
    f = tf.function(mod)
    inputs = tf.TensorSpec([8, None, None, 3], tf.float32)
    cf = f.get_concrete_function(inputs)
    self.assertEqual(cf.outputs[0].shape.as_list(), [8, None])
    flat = cf(tf.ones([8, 1, 2, 3]))
    self.assertEqual(flat.shape, [8, 1 * 2 * 3])

  @parameterized.parameters(1, 2, 3, 4)
  def testPreserveDimsOk(self, preserve_dims):
    in_shape = [10, 2, 3, 4]
    inputs = tf.ones(in_shape)
    mod = reshape.Flatten(preserve_dims=preserve_dims)
    output = mod(inputs)
    flattened_shape = (
        in_shape[:preserve_dims] +
        [np.prod(in_shape[preserve_dims:], dtype=int)])
    self.assertEqual(output.shape, flattened_shape)

  @parameterized.parameters(5, 6, 7, 10)
  def testPreserveDimsError(self, preserve_dims):
    in_shape = [10, 2, 3, 4]
    inputs = tf.ones(in_shape)
    mod = reshape.Flatten(preserve_dims=preserve_dims)
    with self.assertRaisesRegex(ValueError, "Input tensor has 4 dimensions"):
      _ = mod(inputs)

  def testFlattenWithZeroDim(self):
    inputs = tf.ones([1, 0])
    output = reshape.Flatten()(inputs)
    self.assertEqual(output.shape, [1, 0])

  def testInvalidFlattenFromError(self):
    with self.assertRaisesRegex(ValueError, "preserve_dims"):
      reshape.Flatten(preserve_dims=0)

  def testBuildDimError(self):
    mod = reshape.Flatten(preserve_dims=2)
    input_tensor = tf.ones([50])
    with self.assertRaisesRegex(ValueError, "should have at least as many as"):
      mod(input_tensor)

  @parameterized.parameters([1, 8])
  def testReverse(self, batch_size):
    mod = reshape.Flatten(preserve_dims=4)
    inputs = tf.ones([batch_size, 5, 84, 84, 3, 2])
    output = mod(inputs)
    self.assertEqual(output.shape, inputs.shape.as_list()[:4] + [6])
    mod_r = mod.reversed()
    output_r = mod_r(output)
    self.assertEqual(output_r.shape, inputs.shape)


if __name__ == "__main__":
  tf.test.main()
