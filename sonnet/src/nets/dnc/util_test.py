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
"""Tests for sonnet.v2.src.nets.sdnc.util."""

from absl.testing import parameterized
import numpy as np
from sonnet.src import linear
from sonnet.src import test_utils
from sonnet.src.nets.dnc import util
import tensorflow as tf
import tree


class SegmentDimTest(test_utils.TestCase, parameterized.TestCase):

  @parameterized.parameters(([2], [7]), ([], [7]), ([2], []), ([2], [7, 11]),
                            ([2, 11], [7]))
  def testShape(self, initial_shape, final_shape):
    first_shape = tf.TensorShape([3, 3])
    second_shape = tf.TensorShape([5])
    segment_shapes = [first_shape, second_shape]

    inputs_shape = (
        initial_shape +
        [first_shape.num_elements() + second_shape.num_elements()] +
        final_shape)

    inputs = tf.random.uniform(inputs_shape)
    first, second = util.segment_dim(
        inputs, dim=len(initial_shape), shapes=segment_shapes)
    self.assertAllEqual(first.shape.as_list(),
                        initial_shape + first_shape.as_list() + final_shape)
    self.assertAllEqual(second.shape.as_list(),
                        initial_shape + second_shape.as_list() + final_shape)

  @parameterized.parameters(([2], [7]), ([], [7]), ([2], []), ([2], [7, 11]),
                            ([2, 11], [7]))
  def testShapeNegative(self, initial_shape, final_shape):
    first_shape = tf.TensorShape([3, 3])
    second_shape = tf.TensorShape([5])
    segment_shapes = [first_shape, second_shape]

    inputs_shape = (
        initial_shape +
        [first_shape.num_elements() + second_shape.num_elements()] +
        final_shape)

    inputs = tf.random.uniform(inputs_shape)
    first, second = util.segment_dim(
        inputs, dim=-len(final_shape) - 1, shapes=segment_shapes)
    self.assertAllEqual(first.shape.as_list(),
                        initial_shape + first_shape.as_list() + final_shape)
    self.assertAllEqual(second.shape.as_list(),
                        initial_shape + second_shape.as_list() + final_shape)

  def testValues(self):
    segment_shapes = [tf.TensorShape([2]), tf.TensorShape([3])]
    inputs = tf.constant(
        np.hstack([np.zeros((5, 2)), np.ones((5, 3))]), dtype=tf.float32)
    first, second = util.segment_dim(inputs, dim=1, shapes=segment_shapes)

    self.assertAllEqual(first.numpy(), np.zeros_like(first))
    self.assertAllEqual(second.numpy(), np.ones_like(second))

  def testInvalidDims(self):
    segment_shapes = [tf.TensorShape([3]), tf.TensorShape([2])]
    inputs = tf.random.uniform([5, 5])
    with self.assertRaisesRegex(ValueError, 'Invalid dims'):
      util.segment_dim(inputs, 3, segment_shapes)


class BatchInvertPermutationTest(test_utils.TestCase):

  def testCorrectOutput(self):
    # Tests that the _batch_invert_permutation function correctly inverts a
    # batch of permutations.
    batch_size = 5
    length = 7

    permutations = np.empty([batch_size, length], dtype=int)
    for i in range(batch_size):
      permutations[i] = np.random.permutation(length)

    inverse = util.batch_invert_permutation(tf.constant(permutations, tf.int32))

    inverse_np = inverse.numpy()
    for i in range(batch_size):
      for j in range(length):
        self.assertEqual(permutations[i][inverse_np[i][j]], j)


class BatchGatherTest(test_utils.TestCase):

  def testCorrectOutput(self):
    values = np.array([[3, 1, 4, 1], [5, 9, 2, 6], [5, 3, 5, 7]])
    indices = np.array([[1, 2, 0, 3], [3, 0, 1, 2], [0, 2, 1, 3]])
    target = np.array([[1, 4, 3, 1], [6, 5, 9, 2], [5, 5, 3, 7]])
    result = util.batch_gather(tf.constant(values), tf.constant(indices))
    self.assertAllEqual(target, result)


class LinearTest(test_utils.TestCase, parameterized.TestCase):

  def testLinearOutputOneModule(self):
    batch_size = 4
    input_size = 5
    output_size = 3
    lin_a = linear.Linear(output_size)
    inputs = tf.random.uniform([batch_size, input_size])
    output = util.apply_linear(inputs, lin_a, activation=tf.nn.tanh)

    expected_output = np.tanh(
        np.matmul(inputs.numpy(), lin_a.w.numpy()) + lin_a.b.numpy())
    self.assertAllClose(expected_output, output.numpy(), atol=self.get_atol())

  def testLinearOutputTwoModules(self):
    batch_size = 4
    input_size_a = 5
    input_size_b = 6
    output_size = 3
    lin_a = linear.Linear(output_size, name='lin_a')
    lin_b = linear.Linear(output_size, name='lin_b')
    input_a = tf.random.uniform([batch_size, input_size_a])
    input_b = tf.random.uniform([batch_size, input_size_b])
    output = util.apply_linear((input_a, input_b), (lin_a, lin_b),
                               activation=tf.nn.relu)
    expected_output = np.maximum(
        0, (np.matmul(input_a.numpy(), lin_a.w.numpy()) + lin_a.b.numpy() +
            np.matmul(input_b.numpy(), lin_b.w.numpy()) + lin_b.b.numpy()))
    self.assertAllClose(expected_output, output.numpy(), atol=self.get_atol())

  def testDifferentOutputSizeBreaks(self):
    batch_size = 4
    input_size = 5
    output_size_a = 6
    output_size_b = 3

    lin_a = linear.Linear(output_size_a, name='lin_a')
    lin_b = linear.Linear(output_size_b, name='lin_b')
    input_a = tf.random.uniform([batch_size, input_size])
    input_b = tf.random.uniform([batch_size, input_size])
    with self.assertRaisesIncompatibleShapesError(
        tf.errors.InvalidArgumentError):
      util.apply_linear((input_a, input_b), (lin_a, lin_b))

  @parameterized.parameters(
      {
          'input_sizes': 4,
          'module_hidden_sizes': (2, 3)
      },
      {
          'input_sizes': (5, 7),
          'module_hidden_sizes': 10
      },
  )
  def testNonMatchingStructureBreaks(self, input_sizes, module_hidden_sizes):
    batch_size = 16
    inputs = tree.map_structure(
        lambda size: tf.random.uniform([batch_size, size]), input_sizes)
    modules = tree.map_structure(linear.Linear, module_hidden_sizes)

    with self.assertRaisesRegex(ValueError,
                                'don\'t have the same nested structure'):
      util.apply_linear(inputs, modules)

  @parameterized.parameters(
      # Even when list length matches, len must be 2
      {
          'input_sizes': [10] * 3,
          'module_hidden_sizes': [3] * 3
      },
      {
          'input_sizes': [1],
          'module_hidden_sizes': [4]
      })
  def testListMustBeLengthTwo(self, input_sizes, module_hidden_sizes):
    batch_size = 16
    inputs = tree.map_structure(
        lambda size: tf.random.uniform([batch_size, size]), input_sizes)
    modules = tree.map_structure(linear.Linear, module_hidden_sizes)

    with self.assertRaisesRegex(AssertionError, 'must be length 2'):
      util.apply_linear(inputs, modules)


if __name__ == '__main__':
  tf.test.main()
