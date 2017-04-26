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

# Dependency imports
import numpy as np
import sonnet as snt
from sonnet.testing import parameterized
import tensorflow as tf

from tensorflow.python.ops import variables


class LayerNormTest(parameterized.ParameterizedTestCase, tf.test.TestCase):

  def testConstruct(self):
    inputs = tf.placeholder(tf.float32, shape=[None, 64])

    layer_norm1 = snt.LayerNorm()
    layer_norm1(inputs)

    err = (r"Layer normalization expects inputs of rank 2. "
           r"Got inputs of rank \d.")
    with self.assertRaisesRegexp(snt.Error, err):
      malformed_inputs = tf.placeholder(tf.float32, shape=[None, 64, 1])
      layer_norm2 = snt.LayerNorm()
      layer_norm2(malformed_inputs)

  def testFloat16Error(self):
    inputs = tf.placeholder(tf.float16, shape=[None, 64])
    layer_norm = snt.LayerNorm()

    err = (r"LayerNorm does not support `tf\.float16`, insufficient precision "
           "for calculating sufficient statistics.")
    with self.assertRaisesRegexp(snt.NotSupportedError, err):
      layer_norm(inputs)

  @parameterized.NamedParameters(
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


if __name__ == "__main__":
  tf.test.main()
