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

"""Tests for constrained optimization tools in Sonnet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from absl.testing import parameterized
import mock

from sonnet.python.modules import basic
from sonnet.python.modules import moving_average
from sonnet.python.modules import optimization_constraints
from sonnet.python.modules import scale_gradient


import tensorflow.compat.v1 as tf


class OptimizationConstrainsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters([
      (0.5, 0.5),
      (17.3, 17.3),
      (tf.constant_initializer(3.14), 3.14),
      (tf.ones_initializer(), 1.0)
  ])
  def testLagrangeMultInit(self, initializer, exp_lag_mul):
    cons = optimization_constraints.OptimizationConstraints()
    lhs = tf.zeros_like(1.0)
    rhs = tf.ones_like(1.0)
    cons.add(lhs > rhs, initializer=initializer)()
    l = cons.lagrange_multipliers[0]
    with tf.train.MonitoredSession() as sess:
      lag_mul = sess.run(l)
    self.assertAllClose(lag_mul, exp_lag_mul)

  @mock.patch.object(optimization_constraints, '_parametrize')
  def testRateDefaults(self, mocked_parametrized):
    mocked_parametrized.side_effect = (
        lambda x, rate: scale_gradient.scale_gradient(x, -rate))
    rate = 0.1
    cons = optimization_constraints.OptimizationConstraints(rate=rate)
    lhs = tf.zeros_like(1.0)
    rhs = tf.ones_like(1.0)
    x = cons.add(lhs < rhs)()
    v = tf.all_variables()[0]
    dxdl = tf.gradients(x, v)
    with tf.train.MonitoredSession() as sess:
      grads = sess.run(dxdl)
    self.assertAllClose(grads[0], rate)

  @mock.patch.object(optimization_constraints, '_parametrize')
  def testRateOverrides(self, mocked_parametrized):
    mocked_parametrized.side_effect = (
        lambda x, rate: scale_gradient.scale_gradient(x, -rate))
    rate = 7.3
    cons = optimization_constraints.OptimizationConstraints()
    lhs = tf.zeros_like(1.0)
    rhs = tf.ones_like(1.0)
    x = cons.add(lhs < rhs, rate=rate)()
    v = tf.all_variables()[0]
    dxdl = tf.gradients(x, v)
    with tf.train.MonitoredSession() as sess:
      grads = sess.run(dxdl)
    self.assertAllClose(grads[0], rate)

  def testValidRangeDefaults(self):
    valid_range = (1.0, 2.0)
    cons = optimization_constraints.OptimizationConstraints(
        valid_range=valid_range)
    lhs = tf.zeros_like(1.0)
    rhs = tf.ones_like(1.0)
    cons.add(lhs < rhs, initializer=3.0)()
    with tf.train.MonitoredSession() as sess:
      lag_mul = sess.run(cons.lagrange_multipliers[0])
    self.assertAllClose(lag_mul, valid_range[1])

  def testValidRangeOverrides(self):
    cons = optimization_constraints.OptimizationConstraints()
    lhs = tf.zeros_like(1.0)
    rhs = tf.ones_like(1.0)
    valid_range = (1.0, 2.0)
    cons.add(lhs < rhs, initializer=3.0, valid_range=valid_range)()
    with tf.train.MonitoredSession() as sess:
      lag_mul = sess.run(cons.lagrange_multipliers[0])
    self.assertAllClose(lag_mul, valid_range[1])

  @mock.patch.object(
      optimization_constraints.OptimizationConstraints, 'add_geq')
  @mock.patch.object(
      optimization_constraints.OptimizationConstraints, 'add_leq')
  def testOpIdentification(self, mocked_add_leq, mocked_add_geq):
    calls_to_add_leq = [0]
    def mock_add_leq(*args, **kwargs):
      del args
      del kwargs
      calls_to_add_leq[0] += 1
    mocked_add_leq.side_effect = mock_add_leq

    calls_to_add_geq = [0]
    def mock_add_geq(*args, **kwargs):
      del args
      del kwargs
      calls_to_add_geq[0] += 1
    mocked_add_geq.side_effect = mock_add_geq

    cons = optimization_constraints.OptimizationConstraints()
    lhs = tf.zeros_like(1.0)
    rhs = tf.ones_like(1.0)

    self.assertEqual(calls_to_add_leq[0], 0)
    self.assertEqual(calls_to_add_geq[0], 0)
    cons.add(lhs < rhs)
    self.assertEqual(calls_to_add_leq[0], 1)
    self.assertEqual(calls_to_add_geq[0], 0)
    cons.add(lhs <= rhs)
    self.assertEqual(calls_to_add_leq[0], 2)
    self.assertEqual(calls_to_add_geq[0], 0)
    cons.add(lhs > rhs)
    self.assertEqual(calls_to_add_geq[0], 1)
    self.assertEqual(calls_to_add_leq[0], 2)
    cons.add(lhs >= rhs)
    self.assertEqual(calls_to_add_geq[0], 2)
    self.assertEqual(calls_to_add_leq[0], 2)

  def testMinimalRun(self):
    x = basic.TrainableVariable(
        shape=(), initializers={'w': tf.ones_initializer()})()
    x2 = x ** 2.0
    min_value = 0.5
    constr = optimization_constraints.OptimizationConstraints().add(
        x > min_value)

    self.assertFalse(constr._is_connected)
    loss = moving_average.MovingAverage()(
        x2 + tf.random.normal((), stddev=1.0)) + constr()

    self.assertTrue(constr._is_connected)
    with self.assertRaisesRegexp(ValueError, 'Cannot add further constraints'):
      constr.add(x > min_value)
    with self.assertRaisesRegexp(ValueError, 'Cannot add further constraints'):
      constr.add_geq(x, min_value)
    with self.assertRaisesRegexp(ValueError, 'Cannot add further constraints'):
      constr.add_leq(min_value < x)

    opt = tf.train.AdamOptimizer(1e-2, beta1=0.0)
    update = opt.minimize(loss)
    with tf.control_dependencies([update]):
      x2 = tf.identity(x2)

    with tf.train.MonitoredSession() as sess:
      for _ in range(500):
        v, _ = sess.run([x2, update])
    self.assertAllClose(v, min_value**2)


class ConstrainToRangeTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters([
      (0.5, lambda x: x, [1.0]),
      (1.5, lambda x: x, [0.0]),
      (1.5, lambda x: -x, [-1.0]),
      (-.5, lambda x: x, [1.0]),
      (-.5, lambda x: -x, [0.0]),
  ])
  def testOpWrtGradients(self, init_value, fn, expected_grad):
    x = tf.get_variable(
        name='x', shape=(), initializer=tf.constant_initializer(init_value))
    max_val = 1.0
    min_val = 0.0
    z = optimization_constraints._constrain_to_range(x, min_val, max_val)
    g = tf.gradients(fn(z), x)

    with tf.train.MonitoredSession() as sess:
      grads, clipped_vals = sess.run([g, z])

    self.assertAllEqual(grads, expected_grad)
    self.assertGreaterEqual(clipped_vals, min_val)
    self.assertLessEqual(clipped_vals, max_val)

  def testOpMemoization(self):
    def _run_in_new_graph():
      with tf.Graph().as_default():
        z = optimization_constraints._constrain_to_range(tf.zeros((1,)), 0, 1)
        with tf.train.MonitoredSession() as sess:
          sess.run(z)
    for _ in range(10):
      _run_in_new_graph()


if __name__ == '__main__':
  tf.test.main()
