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

"""Tests for sonnet.python.modules.residual."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import sonnet as snt
import tensorflow as tf


class ResidualTest(tf.test.TestCase):

  def setUp(self):
    super(ResidualTest, self).setUp()
    self.batch_size = 3
    self.in_size = 4

  def testShape(self):
    inputs = tf.placeholder(tf.float32, shape=[self.batch_size, self.in_size])
    linear = snt.Linear(self.in_size)
    residual_wrapper = snt.Residual(linear, name="residual")
    output = residual_wrapper(inputs)
    shape = np.ndarray((self.batch_size, self.in_size))

    self.assertShapeEqual(shape, output)

  def testComputation(self):
    inputs = tf.placeholder(tf.float32, shape=[self.batch_size, self.in_size])
    linear = snt.Linear(name="rnn", output_size=self.in_size)
    residual = snt.Residual(linear, name="residual")

    output = residual(inputs)
    w = linear.w
    b = linear.b
    with self.test_session() as sess:
      # With random data, check the TF calculation matches the Numpy version.
      input_data = np.random.randn(self.batch_size, self.in_size)
      tf.global_variables_initializer().run()

      fetches = [output, w, b]
      output = sess.run(fetches, {inputs: input_data})
    output_v, w_v, b_v = output

    output = np.dot(input_data, w_v) + b_v
    residual_output = output + input_data

    self.assertAllClose(residual_output, output_v)


class ResidualCoreTest(tf.test.TestCase):

  def setUp(self):
    super(ResidualCoreTest, self).setUp()
    self.batch_size = 3
    self.in_size = 4

  def testShape(self):
    inputs = tf.placeholder(tf.float32, shape=[self.batch_size, self.in_size])
    prev_state = tf.placeholder(
        tf.float32, shape=[self.batch_size, self.in_size])
    vanilla_rnn = snt.VanillaRNN(self.in_size)
    residual_wrapper = snt.ResidualCore(vanilla_rnn, name="residual")
    output, next_state = residual_wrapper(inputs, prev_state)
    shape = np.ndarray((self.batch_size, self.in_size))

    self.assertEqual(self.in_size, residual_wrapper.output_size)
    self.assertShapeEqual(shape, output)
    self.assertShapeEqual(shape, next_state)

  def testComputation(self):
    inputs = tf.placeholder(tf.float32, shape=[self.batch_size, self.in_size])
    prev_state = tf.placeholder(tf.float32,
                                shape=[self.batch_size, self.in_size])

    vanilla_rnn = snt.VanillaRNN(name="rnn", hidden_size=self.in_size)
    residual = snt.ResidualCore(vanilla_rnn, name="residual")

    output, new_state = residual(inputs, prev_state)
    in_to_hid = vanilla_rnn.in_to_hidden_variables
    hid_to_hid = vanilla_rnn.hidden_to_hidden_variables
    with self.test_session() as sess:
      # With random data, check the TF calculation matches the Numpy version.
      input_data = np.random.randn(self.batch_size, self.in_size)
      prev_state_data = np.random.randn(self.batch_size, self.in_size)
      tf.global_variables_initializer().run()

      fetches = [output, new_state, in_to_hid[0], in_to_hid[1],
                 hid_to_hid[0], hid_to_hid[1]]
      output = sess.run(fetches,
                        {inputs: input_data, prev_state: prev_state_data})
    output_v, new_state_v, in_to_hid_w, in_to_hid_b = output[:4]
    hid_to_hid_w, hid_to_hid_b = output[4:]

    real_in_to_hid = np.dot(input_data, in_to_hid_w) + in_to_hid_b
    real_hid_to_hid = np.dot(prev_state_data, hid_to_hid_w) + hid_to_hid_b
    vanilla_output = np.tanh(real_in_to_hid + real_hid_to_hid)
    residual_output = vanilla_output + input_data

    self.assertAllClose(residual_output, output_v)
    self.assertAllClose(vanilla_output, new_state_v)


class SkipConnectionCoreTest(tf.test.TestCase):

  def setUp(self):
    super(SkipConnectionCoreTest, self).setUp()
    self.batch_size = 3
    self.in_size = 4
    self.hidden_size = 18

  def testOutputSize(self):
    inputs = tf.placeholder(tf.float32, shape=[self.batch_size, self.in_size])
    prev_state = tf.placeholder(
        tf.float32, shape=[self.batch_size, self.hidden_size])
    vanilla_rnn = snt.VanillaRNN(self.hidden_size)
    skip_wrapper = snt.SkipConnectionCore(vanilla_rnn, name="skip")

    with self.assertRaises(ValueError):
      _ = skip_wrapper.output_size

    skip_wrapper(inputs, prev_state)
    self.assertAllEqual([self.in_size + self.hidden_size],
                        skip_wrapper.output_size.as_list())

    skip_wrapper = snt.SkipConnectionCore(
        vanilla_rnn, input_shape=(self.in_size,), name="skip")
    self.assertAllEqual([self.in_size + self.hidden_size],
                        skip_wrapper.output_size.as_list())

  def testShape(self):
    inputs = tf.placeholder(tf.float32, shape=[self.batch_size, self.in_size])
    prev_state = tf.placeholder(
        tf.float32, shape=[self.batch_size, self.hidden_size])
    vanilla_rnn = snt.VanillaRNN(self.hidden_size)
    skip_wrapper = snt.SkipConnectionCore(vanilla_rnn, name="skip")
    output, next_state = skip_wrapper(inputs, prev_state)
    output_shape = np.ndarray((self.batch_size,
                               self.in_size + self.hidden_size))
    state_shape = np.ndarray((self.batch_size, self.hidden_size))

    self.assertShapeEqual(output_shape, output)
    self.assertShapeEqual(state_shape, next_state)

  def testComputation(self):
    inputs = tf.placeholder(tf.float32, shape=[self.batch_size, self.in_size])
    prev_state = tf.placeholder(tf.float32,
                                shape=[self.batch_size, self.in_size])

    vanilla_rnn = snt.VanillaRNN(name="rnn", hidden_size=self.in_size)
    residual = snt.SkipConnectionCore(vanilla_rnn, name="skip")

    output, new_state = residual(inputs, prev_state)
    in_to_hid = vanilla_rnn.in_to_hidden_variables
    hid_to_hid = vanilla_rnn.hidden_to_hidden_variables
    with self.test_session() as sess:
      # With random data, check the TF calculation matches the Numpy version.
      input_data = np.random.randn(self.batch_size, self.in_size)
      prev_state_data = np.random.randn(self.batch_size, self.in_size)
      tf.global_variables_initializer().run()

      fetches = [output, new_state, in_to_hid[0], in_to_hid[1],
                 hid_to_hid[0], hid_to_hid[1]]
      output = sess.run(fetches,
                        {inputs: input_data, prev_state: prev_state_data})
    output_v, new_state_v, in_to_hid_w, in_to_hid_b = output[:4]
    hid_to_hid_w, hid_to_hid_b = output[4:]

    real_in_to_hid = np.dot(input_data, in_to_hid_w) + in_to_hid_b
    real_hid_to_hid = np.dot(prev_state_data, hid_to_hid_w) + hid_to_hid_b
    vanilla_output = np.tanh(real_in_to_hid + real_hid_to_hid)
    skip_output = np.concatenate((input_data, vanilla_output), -1)

    self.assertAllClose(skip_output, output_v)
    self.assertAllClose(vanilla_output, new_state_v)


if __name__ == "__main__":
  tf.test.main()
