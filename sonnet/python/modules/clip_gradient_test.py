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

  def testOpClipDifferentClipValues(self):
    x = tf.placeholder(tf.float32, shape=[2, 1])
    y_1 = snt.clip_gradient(x, 1, 2)
    y_2 = snt.clip_gradient(x, 2, 3)
    z_1 = tf.reduce_sum(y_1 * y_1)
    z_2 = tf.reduce_sum(y_2 * y_2)
    dzdy_1 = tf.gradients(z_1, y_1)[0]
    dzdy_2 = tf.gradients(z_2, y_2)[0]
    dzdx_1 = tf.gradients(z_1, x)[0]
    dzdx_2 = tf.gradients(z_2, x)[0]

    x_np = np.array([[0.5], [2]])
    with self.test_session() as sess:
      y_np_1, dzdy_np_1, dzdx_np_1, y_np_2, dzdy_np_2, dzdx_np_2 = sess.run(
          [y_1, dzdy_1, dzdx_1, y_2, dzdy_2, dzdx_2], feed_dict={x: x_np})

      self.assertAllEqual(y_np_1, x_np)
      self.assertAllEqual(y_np_2, x_np)
      # We do not expect the gradients with respect to the output to be clipped.
      self.assertAllEqual(dzdy_np_1, np.array([[1], [4]]))
      self.assertAllEqual(dzdy_np_2, np.array([[1], [4]]))
      # We expect the gradients w.r.t. the input to be clipped [1, 2] or [2, 3].
      self.assertAllEqual(dzdx_np_1, np.array([[1], [2]]))
      self.assertAllEqual(dzdx_np_2, np.array([[2], [3]]))

  def testOpClipDifferentDtypes(self):
    x_1 = tf.placeholder(tf.float16, shape=())
    snt.clip_gradient(x_1, 0, 1)

    # clip_gradient throws here if the Defun func_name does not use the dtype.
    x_2 = tf.placeholder(tf.float32, shape=())
    snt.clip_gradient(x_2, 0, 1)

  def testShape(self):
    x = tf.placeholder(tf.float32, [None, 10, 13])
    y = snt.clip_gradient(x, 0, 1)
    z = tf.reduce_sum(y * y)
    dzdx = tf.gradients(z, x)[0]

    self.assertAllEqual(y.get_shape().as_list(), [None, 10, 13])
    self.assertAllEqual(dzdx.get_shape().as_list(), [None, 10, 13])


if __name__ == "__main__":
  tf.test.main()
