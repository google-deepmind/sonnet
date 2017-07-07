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

"""Tests for Restore initializer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import re

import numpy as np

from sonnet.python.modules import conv
from sonnet.python.modules.nets import convnet
from sonnet.python.ops import initializers

import tensorflow as tf


def _checkpoint():
  # Delay access to FLAGS.test_srcdir
  return os.path.join(os.environ['TEST_SRCDIR'],
                      'sonnet/sonnet/python/ops/testdata',
                      'restore_initializer_test_checkpoint')


_ONE_CONV_LAYER = 75.901642
_TWO_CONV_LAYERS = 687.9700928
_TWO_CONV_LAYERS_RELU = 61.4554138
_TOLERANCE = 0.001


class RestoreInitializerTest(tf.test.TestCase):

  def testSimpleRestore(self):
    with tf.variable_scope('agent/conv_net_2d/conv_2d_0'):
      bias = tf.get_variable(
          'b',
          shape=[16],
          initializer=initializers.restore_initializer(_checkpoint(), 'b'))
    with self.test_session() as session:
      session.run(tf.global_variables_initializer())
      b = session.run(bias)
      self.assertAllClose(np.linalg.norm(b), 3.9685926, atol=_TOLERANCE)

  def testScopeRestore(self):
    c1 = conv.Conv2D(
        16,
        8,
        4,
        name='conv_2d_0',
        padding=conv.VALID,
        initializers={
            'w':
                initializers.restore_initializer(
                    _checkpoint(), 'w', scope='agent/conv_net_2d/conv_2d_0'),
            'b':
                initializers.restore_initializer(
                    _checkpoint(), 'b', scope='agent/conv_net_2d/conv_2d_0')
        })

    inputs = tf.constant(1 / 255.0, shape=[1, 86, 86, 3])
    outputs = c1(inputs)
    init = tf.global_variables_initializer()
    tf.get_default_graph().finalize()
    with self.test_session() as session:
      session.run(init)
      o = session.run(outputs)

    self.assertAllClose(np.linalg.norm(o), _ONE_CONV_LAYER, atol=_TOLERANCE)

  def testMultipleRestore(self):
    g = tf.Graph()

    restore_initializers = {
        'w': initializers.restore_initializer(_checkpoint(), 'w'),
        'b': initializers.restore_initializer(_checkpoint(), 'b')
    }

    with g.as_default():
      with tf.variable_scope('agent/conv_net_2d'):
        c1 = conv.Conv2D(
            16,
            8,
            4,
            name='conv_2d_0',
            padding=conv.VALID,
            initializers=restore_initializers)
        c2 = conv.Conv2D(
            32,
            4,
            2,
            name='conv_2d_1',
            padding=conv.VALID,
            initializers=restore_initializers)

      inputs = tf.constant(1 / 255.0, shape=[1, 86, 86, 3])
      intermediate_1 = c1(inputs)
      intermediate_2 = c2(tf.nn.relu(intermediate_1))
      outputs = tf.nn.relu(intermediate_2)
      init = tf.global_variables_initializer()

      tf.get_default_graph().finalize()
      with self.test_session() as session:
        session.run(init)
        i1, i2, o = session.run([intermediate_1, intermediate_2, outputs])

      self.assertAllClose(np.linalg.norm(i1), _ONE_CONV_LAYER, atol=_TOLERANCE)
      self.assertAllClose(np.linalg.norm(i2), _TWO_CONV_LAYERS, atol=_TOLERANCE)
      self.assertAllClose(
          np.linalg.norm(o), _TWO_CONV_LAYERS_RELU, atol=_TOLERANCE)

  def testMoreMultipleRestore(self):
    restore_initializers = {
        'w': initializers.restore_initializer(_checkpoint(), 'w'),
        'b': initializers.restore_initializer(_checkpoint(), 'b')
    }

    with tf.variable_scope('agent'):
      c = convnet.ConvNet2D(
          output_channels=(16, 32),
          kernel_shapes=(8, 4),
          strides=(4, 2),
          paddings=[conv.VALID],
          activation=tf.nn.relu,
          activate_final=True,
          initializers=restore_initializers)

    inputs = tf.constant(1 / 255.0, shape=[1, 86, 86, 3])
    outputs = c(inputs)
    init = tf.global_variables_initializer()
    tf.get_default_graph().finalize()
    with self.test_session() as session:
      session.run(init)
      o = session.run(outputs)

    self.assertAllClose(
        np.linalg.norm(o), _TWO_CONV_LAYERS_RELU, atol=_TOLERANCE)

  def testFromDifferentScope(self):
    sub = functools.partial(re.sub, r'^[^/]+/', 'agent/')
    restore_initializers = {
        'w': initializers.restore_initializer(_checkpoint(), 'w', sub),
        'b': initializers.restore_initializer(_checkpoint(), 'b', sub)
    }

    with tf.variable_scope('some_random_scope'):
      c = convnet.ConvNet2D(
          output_channels=(16, 32),
          kernel_shapes=(8, 4),
          strides=(4, 2),
          paddings=[conv.VALID],
          activation=tf.nn.relu,
          activate_final=True,
          initializers=restore_initializers)

    inputs = tf.constant(1 / 255.0, shape=[1, 86, 86, 3])
    outputs = c(inputs)
    init = tf.global_variables_initializer()
    tf.get_default_graph().finalize()
    with self.test_session() as session:
      session.run(init)
      o = session.run(outputs)

    self.assertAllClose(
        np.linalg.norm(o), _TWO_CONV_LAYERS_RELU, atol=_TOLERANCE)

  def testPartitionedVariable(self):
    save_path = os.path.join(self.get_temp_dir(), 'partitioned_variable')
    var_name = 'my_partitioned_var'

    g1 = tf.Graph()
    with g1.as_default():

      def initializer1(shape, dtype, partition_info):
        _ = partition_info  # Not used for creation.
        return tf.constant(True, dtype, shape)

      partitioned_var1 = tf.create_partitioned_variables(
          [1 << 3, 10], [4, 1], initializer1, dtype=tf.bool, name=var_name)

      with self.test_session(graph=g1) as session:
        with tf.device('/cpu:0'):
          tf.global_variables_initializer().run()
          pv1 = session.run(partitioned_var1)
          save = tf.train.Saver(partitioned_var1)
          save.save(session, save_path)

    g2 = tf.Graph()
    with g2.as_default():
      initializer2 = initializers.restore_initializer(save_path, var_name, '')
      partitioned_var2 = tf.create_partitioned_variables(
          [1 << 3, 10], [4, 1], initializer2, dtype=tf.bool, name=var_name)
      with self.test_session(graph=g2) as session:
        tf.global_variables_initializer().run()
        pv2 = session.run(partitioned_var2)

    self.assertAllEqual(pv1, pv2)

  def testTopLevelVariable(self):
    save_path = os.path.join(self.get_temp_dir(), 'toplevel_variable')

    g1 = tf.Graph()
    g2 = tf.Graph()
    with g1.as_default():
      var1 = tf.get_variable(
          'var1', shape=[], initializer=tf.constant_initializer(42))
    with g2.as_default():
      var2 = tf.get_variable(
          'var2',
          shape=[],
          initializer=initializers.restore_initializer(save_path, 'var1'))

    with self.test_session(graph=g1) as session:
      tf.global_variables_initializer().run()
      save = tf.train.Saver([var1])
      save.save(session, save_path)

    with self.test_session(graph=g2) as session:
      tf.global_variables_initializer().run()
      v2 = session.run(var2)

    self.assertAllEqual(v2, 42)


if __name__ == '__main__':
  tf.test.main()
