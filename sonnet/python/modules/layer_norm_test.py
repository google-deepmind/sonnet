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

"""Tests for sonnet.python.modules.layer_norm."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import operator

# Dependency imports
from absl.testing import parameterized
import numpy as np
import sonnet as snt
import tensorflow as tf

from tensorflow.python.ops import variables


def _get_layer_norm_stats(data, axes):
  """Returns mean and variances calculated over the given axes of the data."""

  if axes is None:
    axes = list(range(1, data.ndim))

  # Transpose to put all the normalized dimensions at the end. Well done tharley
  # for the one-liner. For 5D data, and example axes [1, 3] produces transpose
  # arg of [0, 2, 4, 1, 3] which puts all the normalization axes at the end,
  # suitable for flattening down to calculate statistics.
  transposed_data = np.transpose(
      data,
      sorted(set(range(data.ndim)) - set(axes)) + axes)

  # Combine the sizes of all the (now trailing) normalized_dimensions
  normalized_dims_total_size = functools.reduce(
      operator.mul, (data.shape[ax] for ax in axes))

  # Do the reshape - all the non-normalized dimensions are combined by "-1"
  reshaped = np.reshape(transposed_data, [-1, normalized_dims_total_size])

  # Return stats - should be very close to standard normal.
  return {
      "mean": np.mean(reshaped, axis=1),
      "std": np.std(reshaped, axis=1),
  }


class LayerNormTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      ("Float16", tf.float16),
      ("BFloat16", tf.bfloat16),
  )
  def test16BitError(self, dtype):
    inputs = tf.placeholder(dtype, shape=[None, 64])
    layer_norm = snt.LayerNorm()

    err = (r"LayerNorm does not support `tf\.float16` or `tf\.bfloat16`, "
           "insufficient precision for calculating sufficient statistics.")
    with self.assertRaisesRegexp(snt.NotSupportedError, err):
      layer_norm(inputs)

  @parameterized.named_parameters(
      ("Float32", tf.float32),
      ("Float64", tf.float64),
  )
  def testDataType(self, dtype):
    inputs = tf.placeholder(dtype, shape=[None, 64])
    layer_norm = snt.LayerNorm()
    output = layer_norm(inputs)

    self.assertEqual(dtype, output.dtype)
    self.assertEqual(dtype, layer_norm.gamma.dtype.base_dtype)
    self.assertEqual(dtype, layer_norm.beta.dtype.base_dtype)

  def testNormalization(self):
    """Check that inputs are approximately centered and scaled."""
    inputs = tf.constant([[1., 2., 3.], [6., 4., 7.]], dtype=tf.float32)
    ln = snt.LayerNorm()
    outputs = ln(inputs)

    init = tf.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(init)

      outputs_ = sess.run(outputs)

      self.assertAllClose(outputs_.mean(axis=1), [0., 0.], atol=1e-04)
      self.assertAllClose(outputs_.var(axis=1), [1., 1.], atol=1e-04)

  def testSharing(self):
    """Check that the correct number of variables are made when sharing."""

    inputs1 = tf.placeholder(tf.float32, shape=[None, 64])
    inputs2 = tf.placeholder(tf.float32, shape=[None, 64])

    ln = snt.LayerNorm()

    ln(inputs1)
    ln(inputs2)

    self.assertEqual(len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)), 2)

  def testInvalidInitializerParameters(self):
    with self.assertRaisesRegexp(KeyError, "Invalid initializer keys.*"):
      snt.LayerNorm(
          initializers={"not_gamma": tf.contrib.layers.l1_regularizer(0.5)})

    err = "Initializer for 'gamma' is not a callable function"
    with self.assertRaisesRegexp(TypeError, err):
      snt.LayerNorm(initializers={"gamma": tf.zeros([1, 2, 3])})

  def testInvalidPartitionerParameters(self):
    with self.assertRaisesRegexp(KeyError, "Invalid partitioner keys.*"):
      snt.LayerNorm(
          partitioners={"not_gamma": tf.contrib.layers.l1_regularizer(0.5)})

    err = "Partitioner for 'gamma' is not a callable function"
    with self.assertRaisesRegexp(TypeError, err):
      snt.LayerNorm(partitioners={"gamma": tf.zeros([1, 2, 3])})

  def testInvalidRegularizationParameters(self):
    with self.assertRaisesRegexp(KeyError, "Invalid regularizer keys.*"):
      snt.LayerNorm(
          regularizers={"not_gamma": tf.contrib.layers.l1_regularizer(0.5)})

    err = "Regularizer for 'gamma' is not a callable function"
    with self.assertRaisesRegexp(TypeError, err):
      snt.LayerNorm(regularizers={"gamma": tf.zeros([1, 2, 3])})

  def testInitializers(self):
    initializers = {
        "gamma": tf.constant_initializer(2.0),
        "beta": tf.constant_initializer(3.0),
    }

    inputs = tf.placeholder(tf.float32, shape=[None, 10])
    ln = snt.LayerNorm(initializers=initializers)
    self.assertEqual(ln.initializers, initializers)
    ln(inputs)

    init = tf.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(init)

      ones_v = np.ones([inputs.get_shape()[-1]])
      self.assertAllClose(ln.beta.eval(), ones_v * 3.0)
      self.assertAllClose(ln.gamma.eval(), ones_v * 2.0)

  def testRegularizersInRegularizationLosses(self):
    regularizers = {
        "gamma": tf.contrib.layers.l1_regularizer(scale=0.5),
        "beta": tf.contrib.layers.l2_regularizer(scale=0.5),
    }

    inputs = tf.placeholder(tf.float32, shape=[None, 10])
    ln = snt.LayerNorm(regularizers=regularizers)
    self.assertEqual(ln.regularizers, regularizers)
    ln(inputs)

    graph_regularizers = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    self.assertRegexpMatches(graph_regularizers[0].name, ".*l1_regularizer.*")
    self.assertRegexpMatches(graph_regularizers[1].name, ".*l2_regularizer.*")

  def testPartitioners(self):
    partitioners = {
        "gamma": tf.fixed_size_partitioner(num_shards=2),
        "beta": tf.fixed_size_partitioner(num_shards=2),
    }

    inputs = tf.placeholder(tf.float32, shape=[None, 10])
    ln = snt.LayerNorm(partitioners=partitioners)
    self.assertEqual(ln.partitioners, partitioners)
    ln(inputs)

    self.assertEqual(type(ln.gamma), variables.PartitionedVariable)
    self.assertEqual(type(ln.beta), variables.PartitionedVariable)

  @parameterized.parameters(
      # Default, sums over all dimensions except batch:
      {"axes": None, "input_shape": [2, 3]},
      {"axes": None, "input_shape": [4, 5, 6]},
      {"axes": None, "input_shape": [12, 13, 14, 15]},
      # Specify a single axes to sum over:
      {"axes": [1], "input_shape": [5, 6, 7]},
      # Sum over all except final dimension - i.e. Instance Norm.
      {"axes": [1, 2], "input_shape": [10, 11, 12, 14]},
      # Sum over non-contiguous dimensions.
      {"axes": [1, 3], "input_shape": [3, 4, 5, 6, 7]},
      )
  def testAxesDefault(self, axes, input_shape):

    inputs = tf.constant(np.random.rand(*input_shape))
    ln = snt.LayerNorm(axes=axes, offset=False, scale=False)
    output = ln(inputs)

    init = tf.global_variables_initializer()
    with self.test_session() as session:
      session.run(init)
      output_np = session.run(output)

    statistics = _get_layer_norm_stats(output_np, axes=axes)
    self.assertAllClose(statistics["mean"],
                        np.zeros_like(statistics["mean"]),
                        atol=2e-3)
    self.assertAllClose(statistics["std"],
                        np.ones_like(statistics["std"]),
                        atol=2e-3)

  @parameterized.parameters(
      {"axes": True},
      {"axes": False},
      {"axes": 4},
      {"axes": [2, "invalid"]})
  def testInvalidAxes(self, axes):
    msg = "axes should be an iterable of ints"
    with self.assertRaisesRegexp(ValueError, msg):
      snt.LayerNorm(axes=axes)

  @parameterized.parameters(
      {"scale": True, "offset": True},
      {"scale": True, "offset": False},
      {"scale": False, "offset": True},
      {"scale": False, "offset": False})
  def testScaleAndOffset(self, scale, offset):
    inputs = tf.random_uniform([2, 4, 6])
    module = snt.LayerNorm(scale=scale, offset=offset)
    _ = module(inputs)
    variables_dict = {v.name: v for v in module.get_variables()}
    if scale:
      self.assertEqual(variables_dict["layer_norm/gamma:0"].shape, (6,))
    else:
      self.assertNotIn("layer_norm/gamma:0", variables_dict)
    if offset:
      self.assertEqual(variables_dict["layer_norm/beta:0"].shape, (6,))
    else:
      self.assertNotIn("layer_norm/beta:0", variables_dict)


if __name__ == "__main__":
  tf.test.main()
