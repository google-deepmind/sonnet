# Copyright 2019 The Sonnet Authors. All Rights Reserved.
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
"""Tests for sonnet.v2.src.nets.dnc.control."""

from absl.testing import parameterized
import numpy as np
from sonnet.src import recurrent
from sonnet.src import test_utils
from sonnet.src.nets.dnc import control
import tensorflow as tf
import tree


class CoreTest(test_utils.TestCase, parameterized.TestCase):

  @parameterized.parameters({'constructor': recurrent.LSTM},
                            {'constructor': recurrent.GRU})
  def testShape(self, constructor):
    batch_size = 2
    hidden_size = 4
    input_size = 3
    inputs = tf.random.uniform([batch_size, input_size])
    rnn = constructor(hidden_size)
    prev_state = rnn.initial_state(batch_size=batch_size)
    output, next_state = rnn(inputs, prev_state)

    tree.map_structure(lambda t1, t2: self.assertEqual(t1.shape, t2.shape),
                       prev_state, next_state)
    self.assertShapeEqual(np.zeros([batch_size, hidden_size]), output)


class FeedForwardTest(test_utils.TestCase):

  def testShape(self):
    batch_size = 2
    hidden_size = 4
    inputs = tf.random.uniform(shape=[batch_size, hidden_size])
    rnn = control.FeedForward(hidden_size)
    prev_state = rnn.initial_state(batch_size=batch_size)
    output, next_state = rnn(inputs, prev_state)

    output_shape = np.ndarray((batch_size, hidden_size))
    state_shape = np.ndarray((batch_size, 1))

    self.assertShapeEqual(output_shape, output)
    self.assertShapeEqual(state_shape, next_state)

  def testValues(self):
    batch_size = 2
    hidden_size = 4
    input_size = 8
    inputs = tf.random.uniform([batch_size, input_size])

    rnn = control.FeedForward(hidden_size, activation=tf.identity)
    prev_state = rnn.initial_state(batch_size=batch_size)
    output, next_state = rnn(inputs, prev_state)

    weight, bias = rnn.linear.w, rnn.linear.b

    expected_output = np.dot(inputs.numpy(), weight.numpy()) + bias.numpy()

    self.assertAllClose(output.numpy(), expected_output, atol=1e-2)
    # State should remain at dummy value.
    self.assertAllClose(prev_state.numpy(), next_state.numpy(), atol=5e-3)


class DeepCore(test_utils.TestCase, parameterized.TestCase):

  @parameterized.parameters({
      'control_name': 'LSTM',
      'num_layers': 1
  }, {
      'control_name': 'LSTM',
      'num_layers': 2
  }, {
      'control_name': 'GRU',
      'num_layers': 1
  }, {
      'control_name': 'GRU',
      'num_layers': 2
  })
  def testShape(self, control_name, num_layers):
    batch_size = 5
    input_size = 3
    hidden_size = 7
    control_config = {'hidden_size': hidden_size}
    inputs = tf.random.uniform([batch_size, input_size])
    rnn = control.deep_core(
        num_layers=num_layers,
        control_name=control_name,
        control_config=control_config)
    prev_state = rnn.initial_state(batch_size=batch_size)
    output, next_state = rnn(inputs, prev_state)

    # The deep_core concatenates the outputs of the individual cores.
    output_shape = np.ndarray((batch_size, num_layers * hidden_size))
    self.assertShapeEqual(output_shape, output)

    tree.map_structure(lambda t1, t2: self.assertEqual(t1.shape, t2.shape),
                       prev_state, next_state)


if __name__ == '__main__':
  tf.test.main()
