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
# ===================================================

"""Tests for differentiable moving average in Sonnet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from absl.testing import parameterized
from six.moves import range
from sonnet.python.modules import moving_average
import tensorflow.compat.v1 as tf


class MovingAverageTest(parameterized.TestCase, tf.test.TestCase):

  def testFirst(self):
    var = tf.Variable(10.0)
    avg = moving_average.MovingAverage(decay=0.9)(var)

    with tf.train.MonitoredSession() as sess:
      avg_value = sess.run(avg)

      # The avg should be equal to the var after only one iteration
      self.assertEqual(avg_value, 10.0)

  def testReset(self):
    val = tf.placeholder(shape=(), dtype=tf.float32)
    module = moving_average.MovingAverage(decay=0.9)
    avg = module(val)
    reset = module.reset()

    with tf.train.MonitoredSession() as sess:
      avg_value = sess.run(avg, feed_dict={val: 10.0})

      # The avg should be equal to the var after only one iteration
      self.assertEqual(avg_value, 10.0)

      sess.run(reset)
      avg_value = sess.run(avg, feed_dict={val: 100.0})

      # The avg should be equal to the var after only one iteration, again
      self.assertEqual(avg_value, 100.0)

  @parameterized.named_parameters(
      ('no_resource_vars', False),
      ('resource_vars', True))
  def testAverage(self, use_resource_vars):
    decay = 0.9
    num_steps = 10
    init_value = 3.14

    with tf.variable_scope('', use_resource=use_resource_vars):
      var = tf.get_variable(
          'var', (), initializer=tf.constant_initializer(init_value))

    avg = moving_average.MovingAverage(decay=decay)(tf.identity(var))
    with tf.control_dependencies([avg]):
      increment = tf.assign_add(var, 1.0)

    with tf.train.MonitoredSession() as sess:
      expected_value = init_value
      x = init_value
      for _ in range(num_steps):
        avg_value, _ = sess.run([avg, increment])
        self.assertNear(avg_value, expected_value, 1e-4)
        x += 1
        expected_value = expected_value * decay + x * (1 - decay)

  def testAssertDecayIsValid(self):
    with self.assertRaisesRegexp(ValueError, 'Decay must be'):
      moving_average.MovingAverage(decay=2.0)

  def testIsDifferentiable(self):
    x = tf.get_variable(name='x', shape=())
    mva = moving_average.MovingAverage(decay=0.99, local=False)
    y = mva(x)
    dydx = tf.gradients(y, x)
    z = mva(2 * x)
    dzdx = tf.gradients(z, x)
    with tf.train.MonitoredSession() as sess:
      df = sess.run([dydx, dzdx])
    self.assertEqual(df[0], [1.0])
    self.assertEqual(df[1], [2.0])

  def testOpMemoization(self):
    def _run_in_new_graph():
      with tf.Graph().as_default():
        x = tf.ones((1,))
        y = tf.zeros((1,))
        z = moving_average._pass_through_gradients(x, y)
        gx = tf.gradients(z, x)
        with tf.train.MonitoredSession() as sess:
          sess.run([z, gx])
    for _ in range(10):
      _run_in_new_graph()

if __name__ == '__main__':
  tf.test.main()
