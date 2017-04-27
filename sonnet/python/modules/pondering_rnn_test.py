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

"""Tests for Pondering Recurrent cores in sonnet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
from sonnet.python.modules import basic_rnn
from sonnet.python.modules import gated_rnn
from sonnet.python.modules import pondering_rnn
from sonnet.python.modules import rnn_core
from sonnet.testing import parameterized

import tensorflow as tf

from tensorflow.python.util import nest


_VALUES_A = [1., np.array([2, 3.5]), np.array([[-1., -1.], [0., 2.]])]
_VALUES_B = [-0.5, np.array([2.25, 3.]), np.array([[1., -1.], [1., -2.]])]


def _build_nested_tensor(values):
  tensor = (tf.constant(values[0], dtype=tf.float32),
            (tf.constant(values[1], dtype=tf.float32),
             tf.constant(values[2], dtype=tf.float32)))
  return tensor


class OutputTupleCore(rnn_core.RNNCore):
  """Dummy core with multiple outputs."""

  @property
  def output_size(self):
    return tf.TensorShape([1]), tf.TensorShape([1])

  @property
  def state_size(self):
    return tf.TensorShape([1])

  def _build(self):
    pass


class Output2DCore(rnn_core.RNNCore):
  """Dummy core with 2D output."""

  @property
  def output_size(self):
    return tf.TensorShape([1, 1])

  @property
  def state_size(self):
    return tf.TensorShape([1])

  def _build(self):
    pass


class ACTCoreTest(tf.test.TestCase, parameterized.ParameterizedTestCase):

  def _test_nested(self, tensor, values_expected):
    with self.test_session() as sess:
      values_out = sess.run(tensor)
    self.assertEqual(2, len(values_out))
    self.assertEqual(2, len(values_out[1]))
    self.assertEqual(values_expected[0], values_out[0])
    self.assertTrue(np.all(np.equal(values_expected[1], values_out[1][0])))
    self.assertTrue(np.all(np.equal(values_expected[2], values_out[1][1])))

  def testNestedAdd(self):
    values_c = [a + b for a, b in zip(_VALUES_A, _VALUES_B)]
    tf_a = _build_nested_tensor(_VALUES_A)
    tf_b = _build_nested_tensor(_VALUES_B)
    tf_add = pondering_rnn._nested_add(tf_a, tf_b)
    self._test_nested(tf_add, values_c)

  def testNestedUnaryMul(self):
    mul_constant = 0.5
    values_mul = [a * mul_constant for a in _VALUES_A]
    tf_a = _build_nested_tensor(_VALUES_A)
    tf_mul = pondering_rnn._nested_unary_mul(
        tf_a, tf.constant(mul_constant, dtype=tf.float32))
    self._test_nested(tf_mul, values_mul)

  def testNestedZerosLike(self):
    zeros = [0., np.array([0., 0.]), np.array([[0., 0.], [0., 0.]])]
    tf_a = _build_nested_tensor(_VALUES_A)
    tf_zeros = pondering_rnn._nested_zeros_like(tf_a)
    self._test_nested(tf_zeros, zeros)

  def _testACT(self, input_size, hidden_size, output_size, seq_len, batch_size,
               core, get_state_for_halting):
    threshold = 0.99
    act = pondering_rnn.ACTCore(
        core, output_size, threshold, get_state_for_halting)
    seq_input = tf.random_uniform(shape=(seq_len, batch_size, input_size))
    initial_state = core.initial_state(batch_size)
    seq_output = tf.nn.dynamic_rnn(
        act, seq_input, time_major=True, initial_state=initial_state)
    for tensor in nest.flatten(seq_output):
      self.assertEqual(seq_input.dtype, tensor.dtype)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      output = sess.run(seq_output)
    (final_out, (iteration, r_t)), final_cumul_state = output
    self.assertEqual((seq_len, batch_size, output_size),
                     final_out.shape)
    self.assertEqual((seq_len, batch_size, 1),
                     iteration.shape)
    self.assertTrue(np.all(iteration == np.floor(iteration)))
    state_shape = get_state_for_halting(initial_state).get_shape().as_list()
    self.assertEqual(tuple(state_shape),
                     get_state_for_halting(final_cumul_state).shape)
    self.assertEqual((seq_len, batch_size, 1), r_t.shape)
    self.assertTrue(np.all(r_t >= 0))
    self.assertTrue(np.all(r_t <= threshold))

  @parameterized.Parameters((13, 11, 7, 3, 5),
                            (3, 3, 3, 1, 5),
                            (1, 1, 1, 1, 1))
  def testACTLSTM(
      self, input_size, hidden_size, output_size, seq_len, batch_size):
    """Tests ACT using an LSTM for the core."""
    lstm = gated_rnn.LSTM(hidden_size)
    def get_hidden_state(state):
      hidden, unused_cell = state
      return hidden
    self._testACT(input_size, hidden_size, output_size, seq_len, batch_size,
                  lstm, get_hidden_state)

  @parameterized.Parameters((13, 11, 7, 3, 5),
                            (3, 3, 3, 1, 5),
                            (1, 1, 1, 1, 1))
  def testACTVanilla(
      self, input_size, hidden_size, output_size, seq_len, batch_size):
    """Tests ACT using an LSTM for the core."""
    vanilla = basic_rnn.VanillaRNN(hidden_size)
    def get_state(state):
      return state
    self._testACT(input_size, hidden_size, output_size, seq_len, batch_size,
                  vanilla, get_state)

  def testOutputTuple(self):
    core = OutputTupleCore(name="output_tuple_core")
    err = "Output of core should be single Tensor."
    with self.assertRaisesRegexp(ValueError, err):
      pondering_rnn.ACTCore(core, 1, 0.99, lambda state: state)

  def testOutput2D(self):
    core = Output2DCore(name="output_2d_core")
    err = "Output of core should be 1D."
    with self.assertRaisesRegexp(ValueError, err):
      pondering_rnn.ACTCore(core, 1, 0.99, lambda state: state)


if __name__ == "__main__":
  tf.test.main()
