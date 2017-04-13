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

"""Tests for snt.scale_gradient.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

# Dependency imports
import sonnet as snt
from sonnet.testing import parameterized
import tensorflow as tf


class ScaleGradientTest(parameterized.ParameterizedTestCase,
                        tf.test.TestCase):

  @parameterized.Parameters(
      *itertools.product(range(6), [0.0, 0.1, 0.9, 1.0])
  )
  def testOpScale(self, x_, scale):
    x = tf.placeholder(tf.float32, [1])
    y = x * x
    y = snt.scale_gradient(y, scale)
    dydx = tf.gradients([y], [x])[0]

    if scale == 0.0:
      self.assertEqual(y.op.type, "StopGradient")
      self.assertIs(dydx, None)
    else:
      if scale == 1.0:
        self.assertEqual(y.op.type, "Identity")
      else:
        self.assertEqual(y.op.type, "ScaleGradient")

      with self.test_session() as sess:
        dydx_, y_ = sess.run([dydx, y], feed_dict={x: [x_]})

        self.assertAlmostEqual(dydx_[0], 2 * scale * x_, places=6)
        self.assertAlmostEqual(y_[0], x_ ** 2, places=6)

  def testTwoOps(self):
    """Tests that the op can be instantiated twice with appropriate results.

    Implementations with inappropriate global registration of gradients will
    fail this test.
    """

    x = tf.placeholder(tf.float32, [1])
    y = x * x
    y = snt.scale_gradient(y, 0.1)
    y = snt.scale_gradient(y, 0.1)
    dydx = tf.gradients([y], [x])[0]

    with self.test_session() as sess:
      dydx_, y_ = sess.run([dydx, y], feed_dict={x: [3.0]})

      self.assertAlmostEqual(dydx_[0], 2 * 0.1**2 * 3.0, places=6)
      self.assertAlmostEqual(y_[0], 3.0 ** 2, places=6)

  def testShape(self):
    x = tf.placeholder(tf.float32, [None, 10, 13])
    y = snt.scale_gradient(x, 0.1)
    shape = tuple(y.get_shape().as_list())
    self.assertEqual(shape, (None, 10, 13))


if __name__ == "__main__":
  tf.test.main()
