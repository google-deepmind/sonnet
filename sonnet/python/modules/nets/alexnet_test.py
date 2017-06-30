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

"""Tests for snt.nets.alexnet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np

import sonnet as snt
from sonnet.testing import parameterized

import tensorflow as tf
from tensorflow.python.ops import variables


class AlexNetTest(parameterized.ParameterizedTestCase,
                  tf.test.TestCase):

  def testCalcMinSize(self):
    """Test the minimum input size calculator."""
    net = snt.nets.AlexNet(mode=snt.nets.AlexNet.MINI)

    self.assertEqual(net._calc_min_size([(None, (3, 1), None)]), 3)
    self.assertEqual(net._calc_min_size([(None, (3, 1), (3, 2))]), 5)
    self.assertEqual(net._calc_min_size([(None, (3, 1), (3, 2)),
                                         (None, (3, 2), (5, 2))]), 25)

  def testModes(self):
    """Test that each mode can be instantiated."""

    modes = [
        snt.nets.AlexNet.FULL,
        snt.nets.AlexNet.HALF,
        snt.nets.AlexNet.MINI,
    ]

    keep_prob = tf.placeholder(tf.float32)

    for mode in modes:
      net = snt.nets.AlexNet(name="net_{}".format(mode), mode=mode)
      input_shape = [None, net._min_size, net._min_size, 3]
      inputs = tf.placeholder(tf.float32, shape=input_shape)
      net(inputs, keep_prob, is_training=True)

  def testBatchNorm(self):
    """Test that batch norm can be instantiated."""

    net = snt.nets.AlexNet(mode=snt.nets.AlexNet.FULL,
                           use_batch_norm=True)
    input_shape = [net._min_size, net._min_size, 3]
    inputs = tf.placeholder(tf.float32, shape=[None] + input_shape)
    output = net(inputs, is_training=True)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(output, feed_dict={inputs: np.random.rand(10, *input_shape)})

    # Check that an error is raised if we don't specify the is_training flag
    err = "is_training flag must be explicitly specified"
    with self.assertRaisesRegexp(ValueError, err):
      net(inputs)

    # Check Tensorflow flags work
    is_training = tf.placeholder(tf.bool)
    test_local_stats = tf.placeholder(tf.bool)
    net(inputs,
        is_training=is_training,
        test_local_stats=test_local_stats)

    # Check Python is_training flag works
    net(inputs, is_training=False, test_local_stats=False)

    # Check that the appropriate moving statistics variables have been created.
    variance_name = "alex_net/batch_norm/moving_variance:0"
    mean_name = "alex_net/batch_norm/moving_mean:0"
    var_names = [var.name for var in tf.global_variables()]
    self.assertIn(variance_name, var_names)
    self.assertIn(mean_name, var_names)

  def testBatchNormConfig(self):
    batch_norm_config = {
        "scale": True,
    }

    model = snt.nets.AlexNet(mode=snt.nets.AlexNet.FULL,
                             use_batch_norm=True,
                             batch_norm_config=batch_norm_config)

    input_to_net = tf.placeholder(tf.float32, shape=(1, 224, 224, 3))

    model(input_to_net, is_training=True)
    model_variables = model.get_variables()

    self.assertEqual(len(model_variables), 7 * 4)

  def testNoDropoutInTesting(self):
    """An exception should be raised if trying to use dropout when testing."""
    net = snt.nets.AlexNet(mode=snt.nets.AlexNet.FULL)
    input_shape = [net._min_size, net._min_size, 3]
    inputs = tf.placeholder(tf.float32, shape=[None] + input_shape)

    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    output = net(inputs, keep_prob, is_training=False)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      with self.assertRaisesRegexp(tf.errors.InvalidArgumentError, "keep_prob"):
        sess.run(output, feed_dict={inputs: np.random.rand(10, *input_shape),
                                    keep_prob: 0.7})
      # No exception if keep_prob=1
      sess.run(output, feed_dict={inputs: np.random.rand(10, *input_shape),
                                  keep_prob: 1.0})

  def testInputTooSmall(self):
    """Check that an error is raised if the input image is too small."""

    keep_prob = tf.placeholder(tf.float32)
    net = snt.nets.AlexNet(mode=snt.nets.AlexNet.FULL)

    input_shape = [None, net._min_size, net._min_size, 1]
    inputs = tf.placeholder(tf.float32, shape=input_shape)
    net(inputs, keep_prob, is_training=True)

    with self.assertRaisesRegexp(snt.IncompatibleShapeError,
                                 "Image shape too small: (.*?, .*?) < .*?"):
      input_shape = [None, net._min_size - 1, net._min_size - 1, 1]
      inputs = tf.placeholder(tf.float32, shape=input_shape)
      net(inputs, keep_prob, is_training=True)

  def testSharing(self):
    """Check that the correct number of variables are made when sharing."""

    net = snt.nets.AlexNet(mode=snt.nets.AlexNet.MINI)
    inputs1 = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
    inputs2 = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
    keep_prob1 = tf.placeholder(tf.float32)
    keep_prob2 = tf.placeholder(tf.float32)

    net(inputs1, keep_prob1, is_training=True)
    net(inputs2, keep_prob2, is_training=True)

    self.assertEqual(len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)),
                     7 * 2)

    model_variables = net.get_variables()
    self.assertEqual(len(model_variables), 7 * 2)

  def testInvalidInitializationParameters(self):
    err = "Invalid initializer keys.*"
    with self.assertRaisesRegexp(KeyError, err):
      snt.nets.AlexNet(
          initializers={"not_w": tf.truncated_normal_initializer(stddev=1.0)})

    err = "Initializer for 'w' is not a callable function"
    with self.assertRaisesRegexp(TypeError, err):
      snt.nets.AlexNet(initializers={"w": tf.zeros([1, 2, 3])})

  def testInvalidRegularizationParameters(self):
    with self.assertRaisesRegexp(KeyError, "Invalid regularizer keys.*"):
      snt.nets.AlexNet(
          regularizers={"not_w": tf.contrib.layers.l1_regularizer(scale=0.5)})

    err = "Regularizer for 'w' is not a callable function"
    with self.assertRaisesRegexp(TypeError, err):
      snt.nets.AlexNet(regularizers={"w": tf.zeros([1, 2, 3])})

  def testRegularizersInRegularizationLosses(self):
    regularizers = {"w": tf.contrib.layers.l1_regularizer(scale=0.5),
                    "b": tf.contrib.layers.l2_regularizer(scale=0.5)}

    alex_net = snt.nets.AlexNet(regularizers=regularizers, name="alexnet1")

    input_shape = [alex_net._min_size, alex_net._min_size, 3]
    inputs = tf.placeholder(tf.float32, shape=[None] + input_shape)
    alex_net(inputs)

    graph_regularizers = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

    alex_net_conv_layers = len(alex_net.conv_modules)
    for i in range(0, 2 * alex_net_conv_layers, 2):
      self.assertRegexpMatches(graph_regularizers[i].name, ".*l1_regularizer.*")
      self.assertRegexpMatches(
          graph_regularizers[i + 1].name, ".*l2_regularizer.*")

  def testInitializers(self):
    initializers = {
        "w": tf.constant_initializer(1.5),
        "b": tf.constant_initializer(2.5),
    }
    alex_net = snt.nets.AlexNet(mode=snt.nets.AlexNet.FULL,
                                initializers=initializers)
    input_shape = [None, alex_net.min_input_size, alex_net.min_input_size, 3]
    inputs = tf.placeholder(dtype=tf.float32, shape=input_shape)
    alex_net(inputs)
    init = tf.global_variables_initializer()

    with self.test_session() as sess:
      sess.run(init)
      for conv_module in alex_net.conv_modules:
        w_v, b_v = sess.run([conv_module.w, conv_module.b])
        self.assertAllClose(w_v, 1.5 * np.ones(w_v.shape))
        self.assertAllClose(b_v, 2.5 * np.ones(b_v.shape))

  def testPartitioners(self):
    partitioners = {
        "w": tf.fixed_size_partitioner(num_shards=2),
        "b": tf.fixed_size_partitioner(num_shards=2),
    }

    alex_net = snt.nets.AlexNet(partitioners=partitioners, name="alexnet1")

    input_shape = [alex_net._min_size, alex_net._min_size, 3]
    inputs = tf.placeholder(tf.float32, shape=[None] + input_shape)
    alex_net(inputs)

    for conv_module in alex_net.conv_modules:
      self.assertEqual(type(conv_module.w), variables.PartitionedVariable)
      self.assertEqual(type(conv_module.b), variables.PartitionedVariable)

    for linear_module in alex_net.linear_modules:
      self.assertEqual(type(linear_module.w), variables.PartitionedVariable)
      self.assertEqual(type(linear_module.b), variables.PartitionedVariable)

  def testErrorHandling(self):
    err = r"`batch_norm_config` must be a mapping, e\.g\. `dict`."
    with self.assertRaisesRegexp(TypeError, err):
      snt.nets.AlexNet(batch_norm_config="not a valid config")

    err = "AlexNet construction mode 'BLAH' not recognised"
    with self.assertRaisesRegexp(snt.Error, err):
      snt.nets.AlexNet(mode="BLAH")

  def testGetLinearModules(self):
    alex_net = snt.nets.AlexNet(mode=snt.nets.AlexNet.FULL)
    input_shape = [None, alex_net.min_input_size, alex_net.min_input_size, 3]
    inputs = tf.placeholder(dtype=tf.float32, shape=input_shape)
    alex_net(inputs)
    for mod in alex_net.linear_modules:
      self.assertEqual(mod.output_size, 4096)

if __name__ == "__main__":
  tf.test.main()
