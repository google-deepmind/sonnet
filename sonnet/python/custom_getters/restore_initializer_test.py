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
"""Tests for sonnet.python.modules.custom_getters.restore_initializer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import sonnet as snt
from sonnet.testing import parameterized
import tensorflow as tf


class RestoreInitializerTest(parameterized.ParameterizedTestCase,
                             tf.test.TestCase):

  def _save_test_checkpoint(self):

    test_dir = tf.test.get_temp_dir()
    checkpoint_dir = os.path.join(test_dir, "test_path")
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint")

    g = tf.Graph()
    with g.as_default():
      net = snt.Linear(10, name="linear1")
      inputs = tf.placeholder(tf.float32, [10, 10])
      net(inputs)

      saver = tf.train.Saver()

      init = tf.global_variables_initializer()

    with self.test_session(graph=g) as sess:
      sess.run(init)
      saver.save(sess, checkpoint_path, global_step=0)
      expected_values = sess.run({"w": net.w, "b": net.b})

    return checkpoint_dir, expected_values

  def testSimpleUsage(self):
    checkpoint_path, expected_values = self._save_test_checkpoint()
    checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)

    g = tf.Graph()
    with g.as_default():
      custom_getter = snt.custom_getters.restore_initializer(
          filename=checkpoint_path)

      with tf.variable_scope("", custom_getter=custom_getter):
        inputs = tf.placeholder(tf.float32, [10, 10])
        lin1 = snt.Linear(10, name="linear1")
        lin1(inputs)

      init = tf.global_variables_initializer()

    with self.test_session(graph=g) as sess:
      sess.run(init)
      w_value, b_value = sess.run([lin1.w, lin1.b])

    self.assertAllClose(expected_values["w"], w_value)
    self.assertAllClose(expected_values["b"], b_value)

  def testNameFn(self):
    checkpoint_path, expected_values = self._save_test_checkpoint()
    checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)

    def name_fn(name):
      return name.replace("linear2", "linear1")

    g = tf.Graph()
    with g.as_default():
      custom_getter = snt.custom_getters.restore_initializer(
          filename=checkpoint_path,
          name_fn=name_fn)

      with tf.variable_scope("", custom_getter=custom_getter):
        inputs = tf.placeholder(tf.float32, [10, 10])
        lin1 = snt.Linear(10, name="linear2")
        lin1(inputs)

      init = tf.global_variables_initializer()

    with self.test_session(graph=g) as sess:
      sess.run(init)
      w_value, b_value = sess.run([lin1.w, lin1.b])

    self.assertAllClose(expected_values["w"], w_value)
    self.assertAllClose(expected_values["b"], b_value)

  def testCollections(self):
    checkpoint_path, expected_values = self._save_test_checkpoint()
    checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)

    g = tf.Graph()
    with g.as_default():
      custom_getter = snt.custom_getters.restore_initializer(
          filename=checkpoint_path,
          collection="blah")

      with tf.variable_scope("", custom_getter=custom_getter):
        inputs = tf.placeholder(tf.float32, [10, 10])
        lin1 = snt.Linear(10, name="linear1")
        lin1(inputs)

        tf.add_to_collection("blah", lin1.w)

      init = tf.global_variables_initializer()

    with self.test_session(graph=g) as sess:
      sess.run(init)
      w_value = sess.run(lin1.w)

    self.assertFalse(np.allclose(expected_values["w"], w_value))
    # b is initialized to zero always.


if __name__ == "__main__":
  tf.test.main()
