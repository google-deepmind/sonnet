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
"""Tests for sonnet.v2.src.batch_apply."""

from absl.testing import parameterized
import numpy as np
from sonnet.src import base
from sonnet.src import batch_apply
from sonnet.src import test_utils
import tensorflow as tf

EXAMPLE_INPUTS = (
    ((1, 2, 3), 1),
    ((1, 2, 3), 2),
    ((1, 2, 3, 4), 3),
    ((1, 2, 3, 4, 5, 6), 4),
)


class BatchApplyTest(test_utils.TestCase):

  def test_simple(self):
    m = batch_apply.BatchApply(AddOne())
    x = tf.zeros([2, 3, 4])
    y = m(x)
    self.assertAllEqual(y, tf.ones([2, 3, 4]))

  def test_no_output(self):
    m = batch_apply.BatchApply(NoOutputModule())
    y = m(tf.ones([1, 1, 1]))
    self.assertIsNone(y)

  def test_kwargs(self):
    m = batch_apply.BatchApply(KwargsModule())
    y = m(tf.ones([1, 1, 1]), is_training=True)
    self.assertIsNone(y)


class MergeLeadingDimsTest(test_utils.TestCase, parameterized.TestCase):

  @parameterized.parameters(object(), (np.ones([]),), 1, None)
  def test_x_not_tensor(self, x):
    self.assertIs(x, batch_apply.merge_leading_dims(x, 1))

  @parameterized.parameters(*EXAMPLE_INPUTS)
  def test_static_shape(self, x_shape, num_dims):
    x = tf.ones(x_shape)
    y = batch_apply.merge_leading_dims(x, num_dims)
    y_shape = (np.prod(x_shape[:num_dims]),) + x_shape[num_dims:]
    self.assertEqual(y.shape, y_shape)

  @parameterized.parameters(*EXAMPLE_INPUTS)
  def test_dynamic_shape(self, x_shape, num_dims):
    merge = tf.function(batch_apply.merge_leading_dims)

    x = tf.TensorSpec([None for _ in x_shape])
    cf = merge.get_concrete_function(x, num_dims)
    y_shape = (np.prod(x_shape[:num_dims]),) + x_shape[num_dims:]
    y_shape_dynamic = cf.output_shapes
    y_shape_dynamic.assert_is_compatible_with(y_shape)

    x = tf.ones(x_shape)
    y = cf(x)
    self.assertEqual(y.shape, y_shape)

  @parameterized.parameters(*EXAMPLE_INPUTS)
  def test_dynamic_shape_has_static_info_in_graph(self, x_shape, num_dims):
    y_shape = (np.prod(x_shape[:num_dims]),) + x_shape[num_dims:]

    @tf.function
    def merge(x, num_dims):
      y = batch_apply.merge_leading_dims(x, num_dims)
      self.assertIsNotNone(y.shape.dims)
      y.shape.assert_is_compatible_with(y_shape)
      # Make sure we have static shape info except for the trailing None.
      self.assertNotIn(None, y.shape[:-1])
      self.assertIsNone(y.shape[-1])
      return y

    # Fill `None` in the last dimension (which won't be merged).
    x = tf.TensorSpec(x_shape[:-1] + (None,))
    cf = merge.get_concrete_function(x, num_dims)
    y_shape_dynamic = cf.output_shapes
    y_shape_dynamic.assert_is_compatible_with(y_shape)


class SplitLeadingDimTest(test_utils.TestCase, parameterized.TestCase):

  @parameterized.parameters(object(), (np.ones([]),), 1, None)
  def test_x_not_tensor(self, x):
    self.assertIs(x, batch_apply.split_leading_dim(x, None, 1))

  @parameterized.parameters(*EXAMPLE_INPUTS)
  def test_static_shape(self, i_shape, num_dims):
    x_shape = (np.prod(i_shape[:num_dims]),) + (2, 2)
    y_shape = i_shape[:num_dims] + x_shape[1:]
    x = tf.ones(x_shape)
    i = tf.ones(i_shape)
    y = batch_apply.split_leading_dim(x, i, num_dims)
    self.assertEqual(y.shape, y_shape)

  @parameterized.parameters(*EXAMPLE_INPUTS)
  def test_dynamic_shape(self, i_shape, num_dims):
    x_shape = (np.prod(i_shape[:num_dims]),) + (2, 2)
    y_shape = i_shape[:num_dims] + x_shape[1:]

    # Build a concrete function with fully dynamic input dimensions.
    x = tf.TensorSpec([None for _ in x_shape])
    i = tf.TensorSpec([None for _ in i_shape])
    split = tf.function(batch_apply.split_leading_dim)
    cf = split.get_concrete_function(x, i, num_dims)
    y_shape_dynamic = cf.output_shapes
    y_shape_dynamic.assert_is_compatible_with(y_shape)

    # Make use of the concrete function with fully specified inputs.
    x = tf.ones(x_shape)
    i = tf.ones(i_shape)
    y = cf(x, i)
    self.assertEqual(y.shape, y_shape)

  @parameterized.parameters(*EXAMPLE_INPUTS)
  def test_dynamic_shape_has_static_info_in_graph(self, i_shape, num_dims):
    x_shape = (np.prod(i_shape[:num_dims]),) + (2, 2)
    y_shape = i_shape[:num_dims] + x_shape[1:]

    @tf.function
    def split(x, i, num_dims):
      y = batch_apply.split_leading_dim(x, i, num_dims)
      self.assertIsNotNone(y.shape.dims)
      y.shape.assert_is_compatible_with(y_shape)
      self.assertNotIn(None, y.shape[1:])
      return y

    # Build a concrete function with fully dynamic input dimensions.
    x = tf.TensorSpec((None,) + x_shape[1:])
    i = tf.TensorSpec((None,) + i_shape[1:])
    cf = split.get_concrete_function(x, i, num_dims)
    y_shape_dynamic = cf.output_shapes
    y_shape_dynamic.assert_is_compatible_with(y_shape)


class NoOutputModule(base.Module):

  def __call__(self, x):
    return None


class KwargsModule(base.Module):

  def __call__(self, x, is_training=None):
    if is_training:
      return None


class AddOne(base.Module):

  def __call__(self, x):
    assert len(x.shape) == 2, "Requires rank 2 input."
    return x + 1.


if __name__ == "__main__":
  tf.test.main()
