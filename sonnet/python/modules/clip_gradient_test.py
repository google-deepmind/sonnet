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

"""Tests for snt.clip_gradient."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
import sonnet as snt
import tensorflow as tf


class ClipGradientTest(tf.test.TestCase):

  def testOpClip(self):
    x = tf.placeholder(tf.float32, shape=[2, 1])
    y = snt.clip_gradient(x, 2, 3)
    z = tf.reduce_sum(y * y)
    dzdy = tf.gradients(z, y)[0]
    dzdx = tf.gradients(z, x)[0]

    x_np = np.array([[0.5], [2]])
    with self.test_session() as sess:
      y_np, dzdy_np, dzdx_np = sess.run([y, dzdy, dzdx], feed_dict={x: x_np})

      self.assertAllEqual(y_np, x_np)
      # We do not expect the gradients with respect to the output to be clipped.
      self.assertAllEqual(dzdy_np, np.array([[1], [4]]))
      # We expect the gradients with respect to the input to be clipped [2, 3].
      self.assertAllEqual(dzdx_np, np.array([[2], [3]]))

  def testShape(self):
    x = tf.placeholder(tf.float32, [None, 10, 13])
    y = snt.clip_gradient(x, 0, 1)
    z = tf.reduce_sum(y * y)
    dzdx = tf.gradients(z, x)[0]

    self.assertAllEqual(y.get_shape().as_list(), [None, 10, 13])
    self.assertAllEqual(dzdx.get_shape().as_list(), [None, 10, 13])


if __name__ == "__main__":
  tf.test.main()
