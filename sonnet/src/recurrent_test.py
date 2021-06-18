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
"""Tests for sonnet.v2.src.recurrent."""

import itertools
import unittest

from absl.testing import parameterized
import numpy as np
from sonnet.src import initializers
from sonnet.src import recurrent
from sonnet.src import test_utils
import tensorflow as tf
import tree


class VanillaRNNTest(test_utils.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.batch_size = 3
    self.input_size = 2
    self.hidden_size = 16

  @parameterized.parameters([False, True])
  def testComputationAgainstNumPy(self, use_tf_function):
    inputs = self.evaluate(
        tf.random.uniform([self.batch_size, self.input_size]))
    core = recurrent.VanillaRNN(
        hidden_size=self.hidden_size, activation=tf.tanh)
    prev_state = self.evaluate(core.initial_state(self.batch_size))

    core_fn = tf.function(core) if use_tf_function else core
    outputs, next_state = core_fn(tf.convert_to_tensor(inputs), prev_state)

    expected_output = np.tanh(
        inputs.dot(self.evaluate(core.input_to_hidden)) +
        prev_state.dot(self.evaluate(core.hidden_to_hidden)) +
        self.evaluate(core._b))

    atol = 3e-2 if self.primary_device == "TPU" else 1e-6
    self.assertAllClose(outputs, expected_output, atol=atol)
    self.assertAllClose(next_state, expected_output, atol=atol)

  def testDtypeMismatch(self):
    core = recurrent.VanillaRNN(hidden_size=self.hidden_size, dtype=tf.bfloat16)
    inputs = tf.random.uniform([self.batch_size, self.input_size])
    prev_state = core.initial_state(self.batch_size)
    self.assertIs(prev_state.dtype, tf.bfloat16)
    with self.assertRaisesRegex(
        TypeError, "inputs must have dtype tf.bfloat16, got tf.float32"):
      core(inputs, prev_state)

  def testInitialization(self):
    core = recurrent.VanillaRNN(
        hidden_size=self.hidden_size,
        w_i_init=initializers.Ones(),
        w_h_init=initializers.Ones(),
        b_init=initializers.Ones())
    inputs = tf.random.uniform([self.batch_size, self.input_size])
    prev_state = core.initial_state(self.batch_size)
    core(inputs, prev_state)

    for v in core.variables:
      self.assertAllClose(self.evaluate(v), self.evaluate(tf.ones_like(v)))


class DeepRNNTest(test_utils.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.batch_size = 3
    self.input_size = 2
    self.hidden_size = 16

  @parameterized.parameters([False, True])
  def testComputationAgainstNumPy(self, use_tf_function):
    inputs = self.evaluate(
        tf.random.uniform([self.batch_size, self.input_size]))
    core = recurrent.DeepRNN([
        recurrent.VanillaRNN(hidden_size=self.hidden_size),
        recurrent.VanillaRNN(hidden_size=2 * self.hidden_size)
    ])
    prev_state = self.evaluate(core.initial_state(self.batch_size))

    core_fn = tf.function(core) if use_tf_function else core
    outputs, next_state = core_fn(tf.convert_to_tensor(inputs), prev_state)

    expected_outputs = inputs
    expected_next_state = list(prev_state)
    for idx, l in enumerate(core._layers):
      expected_outputs, expected_next_state[idx] = l(expected_outputs,
                                                     prev_state[idx])

    self.assertAllClose(outputs, expected_outputs)
    self.assertAllClose(next_state, tuple(expected_next_state))

  @parameterized.parameters([False, True])
  def testComputationAgainstNumPyWithCallables(self, use_tf_function):
    inputs = self.evaluate(
        tf.random.uniform([self.batch_size, self.input_size]))
    core = recurrent.DeepRNN([tf.tanh, tf.sign])
    prev_state = self.evaluate(core.initial_state(self.batch_size))

    core_fn = tf.function(core) if use_tf_function else core
    outputs, next_state = core_fn(tf.convert_to_tensor(inputs), prev_state)

    self.assertAllClose(outputs, np.sign(np.tanh(inputs)))
    self.assertEqual(next_state, prev_state)

  def testInitialState(self):
    core0 = recurrent.VanillaRNN(hidden_size=self.hidden_size)
    core1 = recurrent.VanillaRNN(hidden_size=2 * self.hidden_size)
    deep_rnn = recurrent.DeepRNN([core0, tf.tanh, core1, tf.sign])
    prev_state = deep_rnn.initial_state(self.batch_size)
    self.assertAllClose(prev_state[0], core0.initial_state(self.batch_size))
    self.assertAllClose(prev_state[1], core1.initial_state(self.batch_size))

  @parameterized.parameters([False, True])
  def testWithSkipConnectionsOutputs(self, use_tf_function):
    inputs = self.evaluate(
        tf.random.uniform([self.batch_size, self.input_size]))
    core = recurrent.deep_rnn_with_skip_connections([
        recurrent.VanillaRNN(hidden_size=self.hidden_size),
        recurrent.VanillaRNN(hidden_size=2 * self.hidden_size)
    ],
                                                    concat_final_output=False)
    prev_state = self.evaluate(core.initial_state(self.batch_size))

    core_fn = tf.function(core) if use_tf_function else core
    outputs, _ = core_fn(tf.convert_to_tensor(inputs), prev_state)

    self.assertEqual(outputs.shape,
                     tf.TensorShape([self.batch_size, 2 * self.hidden_size]))

  def testWithConnectionsValidation(self):
    with self.assertRaisesRegex(ValueError, "to be instances of RNNCore"):
      recurrent.deep_rnn_with_skip_connections([tf.tanh])
    with self.assertRaisesRegex(ValueError, "to be instances of RNNCore"):
      recurrent.deep_rnn_with_residual_connections([tf.tanh])


class LSTMTest(test_utils.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.batch_size = 3
    self.input_size = 2
    self.hidden_size = 16

  @parameterized.parameters(
      itertools.product([False, True], [None, 4], [0.0, 1.0]))
  def testComputationAgainstNumPy(self, use_tf_function, projection_size,
                                  forget_bias):
    inputs = self.evaluate(
        tf.random.uniform([self.batch_size, self.input_size]))
    core = recurrent.LSTM(
        self.hidden_size,
        projection_size=projection_size,
        forget_bias=forget_bias)
    prev_state = self.evaluate(core.initial_state(self.batch_size))

    core_fn = tf.function(core) if use_tf_function else core
    outputs, next_state = core_fn(tf.convert_to_tensor(inputs), prev_state)

    w_ii, w_if, w_ig, w_io = np.hsplit(self.evaluate(core.input_to_hidden), 4)
    w_hi, w_hf, w_hg, w_ho = np.hsplit(self.evaluate(core.hidden_to_hidden), 4)
    b_i, b_f, b_g, b_o = np.hsplit(self.evaluate(core.b), 4)
    i = expit(inputs.dot(w_ii) + prev_state.hidden.dot(w_hi) + b_i)
    f = expit(inputs.dot(w_if) + prev_state.hidden.dot(w_hf) + b_f)
    g = np.tanh(inputs.dot(w_ig) + prev_state.hidden.dot(w_hg) + b_g)
    o = expit(inputs.dot(w_io) + prev_state.hidden.dot(w_ho) + b_o)

    expected_cell = f * prev_state.cell + i * g
    expected_hidden = o * np.tanh(expected_cell)

    if projection_size is not None:
      expected_hidden = expected_hidden.dot(self.evaluate(core.projection))

    atol = 1e-2 if self.primary_device == "TPU" else 1e-6
    self.assertAllClose(outputs, next_state.hidden, atol=atol)
    self.assertAllClose(expected_hidden, next_state.hidden, atol=atol)
    self.assertAllClose(expected_cell, next_state.cell, atol=atol)

  def testDtypeMismatch(self):
    core = recurrent.LSTM(hidden_size=self.hidden_size, dtype=tf.bfloat16)
    inputs = tf.random.uniform([self.batch_size, self.input_size])
    prev_state = core.initial_state(self.batch_size)
    self.assertIs(prev_state.hidden.dtype, tf.bfloat16)
    self.assertIs(prev_state.cell.dtype, tf.bfloat16)
    with self.assertRaisesRegex(
        TypeError, "inputs must have dtype tf.bfloat16, got tf.float32"):
      core(inputs, prev_state)

  def testInitialization(self):
    projection_size = 4
    core = recurrent.LSTM(
        hidden_size=self.hidden_size,
        projection_size=projection_size,
        projection_init=initializers.Ones(),
        w_i_init=initializers.Ones(),
        w_h_init=initializers.Ones(),
        b_init=initializers.Ones(),
        forget_bias=0.0)
    inputs = tf.random.uniform([self.batch_size, self.input_size])
    prev_state = core.initial_state(self.batch_size)
    core(inputs, prev_state)

    for v in core.variables:
      self.assertAllClose(self.evaluate(v), self.evaluate(tf.ones_like(v)))

  @parameterized.parameters([1e-6, 0.5, 1 - 1e-6])
  def testRecurrentDropout(self, rate):
    num_steps = 2
    inputs = tf.random.uniform([num_steps, self.batch_size, self.input_size])

    train_core, test_core = recurrent.lstm_with_recurrent_dropout(
        self.hidden_size, dropout=rate)
    [_, train_output
    ], _ = recurrent.dynamic_unroll(train_core, inputs,
                                    train_core.initial_state(self.batch_size))
    [_, test_output
    ], _ = recurrent.dynamic_unroll(test_core, inputs,
                                    test_core.initial_state(self.batch_size))

    almost_zero = rate == 1e-6
    if almost_zero:
      # The train and test versions have the same output when rate is ~0.
      rtol = 1e-3 if self.primary_device == "TPU" else 1e-6
      self.assertAllClose(train_output, test_output, rtol=rtol)
    else:
      self.assertGreater(
          self.evaluate(tf.reduce_max(tf.abs(train_output - test_output))),
          0.001)

  def testRecurrentDropoutInvalid(self):
    with self.assertRaisesRegex(ValueError,
                                r"dropout must be in the range \[0, 1\).+"):
      recurrent.lstm_with_recurrent_dropout(self.hidden_size, -1)


class UnrolledLSTMTest(test_utils.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()

    self.batch_size = 3
    self.input_size = 2
    self.hidden_size = 16

  @parameterized.parameters(itertools.product([1, 4], [True, False]))
  def testComputationAgainstLSTM(self, num_steps, use_tf_function):
    unrolled_lstm = recurrent.UnrolledLSTM(self.hidden_size)
    initial_state = unrolled_lstm.initial_state(self.batch_size)

    if use_tf_function:
      # TODO(b/134377706): remove the wrapper once the bug is fixed.
      # Currently implementation selector requires an explicit device block
      # inside a tf.function to work.
      @tf.function
      def unrolled_lstm_fn(*args, **kwargs):
        with tf.device("/device:{}:0".format(self.primary_device)):
          return unrolled_lstm(*args, **kwargs)
    else:
      unrolled_lstm_fn = unrolled_lstm

    input_sequence = tf.random.uniform(
        [num_steps, self.batch_size, self.input_size])
    output_sequence, final_state = unrolled_lstm_fn(input_sequence,
                                                    initial_state)

    with tf.device("/device:CPU:0"):  # Use CPU as the baseline.
      lstm = recurrent.LSTM(self.hidden_size)
      lstm._initialize(input_sequence[0])
      lstm._w_i = unrolled_lstm._w_i
      lstm._w_h = unrolled_lstm._w_h
      lstm.b = unrolled_lstm.b
      expected_output_sequence, expected_final_state = recurrent.dynamic_unroll(
          lstm, input_sequence, lstm.initial_state(self.batch_size))

    atol = 1e-2 if self.primary_device == "TPU" else 1e-6
    self.assertAllClose(output_sequence, expected_output_sequence, atol=atol)
    self.assertAllClose(
        final_state.hidden, expected_final_state.hidden, atol=atol)
    self.assertAllClose(final_state.cell, expected_final_state.cell, atol=atol)

  @parameterized.parameters([True, False])
  def testNumStepsPolymorphism(self, use_tf_function):
    unrolled_lstm = recurrent.UnrolledLSTM(self.hidden_size)
    initial_state = unrolled_lstm.initial_state(self.batch_size)

    if use_tf_function:
      # TODO(b/134377706): remove the wrapper once the bug is fixed.
      # Currently implementation selector requires an explicit device block
      # inside a tf.function to work.
      @tf.function
      def unrolled_lstm_fn(*args, **kwargs):
        with tf.device("/device:%s:0" % self.primary_device):
          return unrolled_lstm(*args, **kwargs)
    else:
      unrolled_lstm_fn = unrolled_lstm

    # Check that the same instance can be called with different `num_steps`.
    for num_steps in [1, 2, 4]:
      output_sequence, _ = unrolled_lstm_fn(
          tf.random.uniform([num_steps, self.batch_size, self.input_size]),
          initial_state)
      self.assertEqual(output_sequence.shape[0], num_steps)

  def testDtypeMismatch(self):
    unrolled_lstm = recurrent.UnrolledLSTM(
        hidden_size=self.hidden_size, dtype=tf.bfloat16)
    input_sequence = tf.random.uniform([1, self.batch_size, self.input_size])
    initial_state = unrolled_lstm.initial_state(self.batch_size)
    self.assertIs(initial_state.hidden.dtype, tf.bfloat16)
    self.assertIs(initial_state.cell.dtype, tf.bfloat16)
    with self.assertRaisesRegex(
        TypeError, "inputs must have dtype tf.bfloat16, got tf.float32"):
      unrolled_lstm(input_sequence, initial_state)

  def testInitialization(self):
    unrolled_lstm = recurrent.UnrolledLSTM(
        hidden_size=self.hidden_size,
        forget_bias=0.0,
        w_i_init=initializers.Ones(),
        w_h_init=initializers.Ones(),
        b_init=initializers.Ones())
    input_sequence = tf.random.uniform([1, self.batch_size, self.input_size])
    initial_state = unrolled_lstm.initial_state(self.batch_size)
    unrolled_lstm(input_sequence, initial_state)

    for v in unrolled_lstm.variables:
      self.assertAllClose(self.evaluate(v), self.evaluate(tf.ones_like(v)))


class ConvNDLSTMTest(test_utils.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.batch_size = 3
    self.input_size = 2
    self.hidden_size = 16
    self.input_channels = 3
    self.output_channels = 5

  @parameterized.parameters(
      itertools.product(
          [False, True],
          [recurrent.Conv1DLSTM, recurrent.Conv2DLSTM, recurrent.Conv3DLSTM]))
  def testComputationAgainstNumPy(self, use_tf_function, core_cls):
    if core_cls is recurrent.Conv1DLSTM:
      num_spatial_dims = 1
    elif core_cls is recurrent.Conv2DLSTM:
      num_spatial_dims = 2
    else:
      assert core_cls is recurrent.Conv3DLSTM
      num_spatial_dims = 3

    input_shape = ((self.batch_size,) + (self.input_size,) * num_spatial_dims +
                   (self.input_channels,))

    inputs = self.evaluate(tf.random.uniform(input_shape))
    core = core_cls(input_shape[1:], self.output_channels, kernel_shape=1)
    prev_state = self.evaluate(core.initial_state(self.batch_size))

    core_fn = tf.function(core) if use_tf_function else core
    outputs, next_state = core_fn(tf.convert_to_tensor(inputs), prev_state)

    def conv(x, f):
      # NumPy does not have an out-of-the-box alternative.
      return self.evaluate(tf.nn.convolution(x, f, strides=1, padding="SAME"))

    w_i = self.evaluate(core.input_to_hidden)
    w_h = self.evaluate(core.hidden_to_hidden)
    w_ii, w_if, w_ig, w_io = np.split(w_i, 4, axis=-1)
    w_hi, w_hf, w_hg, w_ho = np.split(w_h, 4, axis=-1)
    b_i, b_f, b_g, b_o = np.hsplit(self.evaluate(core.b), 4)
    i = expit(conv(inputs, w_ii) + conv(prev_state.hidden, w_hi) + b_i)
    f = expit(conv(inputs, w_if) + conv(prev_state.hidden, w_hf) + b_f)
    g = np.tanh(conv(inputs, w_ig) + conv(prev_state.hidden, w_hg) + b_g)
    o = expit(conv(inputs, w_io) + conv(prev_state.hidden, w_ho) + b_o)

    expected_cell = f * prev_state.cell + i * g
    expected_hidden = o * np.tanh(expected_cell)

    atol = 1e-2 if self.primary_device == "TPU" else 1e-6
    self.assertAllClose(outputs, next_state.hidden, atol=atol)
    self.assertAllClose(expected_hidden, next_state.hidden, atol=atol)
    self.assertAllClose(expected_cell, next_state.cell, atol=atol)

  def testDtypeMismatch(self):
    num_spatial_dims = 1
    input_shape = ((self.batch_size,) + (self.input_size,) * num_spatial_dims +
                   (self.input_channels,))

    core = recurrent.Conv1DLSTM(
        input_shape[1:],
        self.output_channels,
        kernel_shape=1,
        dtype=tf.bfloat16)
    inputs = tf.random.uniform(input_shape)
    prev_state = core.initial_state(self.batch_size)
    self.assertIs(prev_state.hidden.dtype, tf.bfloat16)
    self.assertIs(prev_state.cell.dtype, tf.bfloat16)
    with self.assertRaisesRegex(
        TypeError, "inputs must have dtype tf.bfloat16, got tf.float32"):
      core(inputs, prev_state)

  def testInitialization(self):
    num_spatial_dims = 1
    input_shape = ((self.batch_size,) + (self.input_size,) * num_spatial_dims +
                   (self.input_channels,))

    inputs = tf.random.uniform(input_shape)
    core = recurrent.Conv1DLSTM(
        input_shape[1:],
        self.output_channels,
        kernel_shape=1,
        forget_bias=0.0,
        w_i_init=initializers.Ones(),
        w_h_init=initializers.Ones(),
        b_init=initializers.Ones())
    prev_state = core.initial_state(self.batch_size)
    core(inputs, prev_state)

    for v in core.variables:
      self.assertAllClose(self.evaluate(v), self.evaluate(tf.ones_like(v)))


class GRUTest(test_utils.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.batch_size = 3
    self.input_size = 2
    self.hidden_size = 16

  @parameterized.parameters([False, True])
  def testComputationAgainstNumPy(self, use_tf_function):
    inputs = self.evaluate(
        tf.random.uniform([self.batch_size, self.input_size]))
    core = recurrent.GRU(self.hidden_size)
    prev_state = self.evaluate(core.initial_state(self.batch_size))

    core_fn = tf.function(core) if use_tf_function else core
    outputs, next_state = core_fn(tf.convert_to_tensor(inputs), prev_state)

    w_iz, w_ir, w_ia = np.hsplit(self.evaluate(core.input_to_hidden), 3)
    w_hz, w_hr, w_ha = np.hsplit(self.evaluate(core.hidden_to_hidden), 3)
    b_z, b_r, b_a = np.hsplit(self.evaluate(core.b), 3)

    z = expit(inputs.dot(w_iz) + prev_state.dot(w_hz) + b_z)
    r = expit(inputs.dot(w_ir) + prev_state.dot(w_hr) + b_r)
    a = np.tanh(inputs.dot(w_ia) + (r * prev_state).dot(w_ha) + b_a)
    expected_state = (1 - z) * prev_state + z * a

    atol = 1e-2 if self.primary_device == "TPU" else 1e-6
    self.assertAllClose(outputs, next_state, atol=atol)
    self.assertAllClose(self.evaluate(next_state), expected_state, atol=atol)

  def testDtypeMismatch(self):
    core = recurrent.GRU(hidden_size=self.hidden_size, dtype=tf.bfloat16)
    inputs = tf.random.uniform([self.batch_size, self.input_size])
    prev_state = core.initial_state(self.batch_size)
    self.assertIs(prev_state.dtype, tf.bfloat16)
    with self.assertRaisesRegex(
        TypeError, "inputs must have dtype tf.bfloat16, got tf.float32"):
      core(inputs, prev_state)

  def testInitialization(self):
    core = recurrent.GRU(
        hidden_size=self.hidden_size,
        w_i_init=initializers.Ones(),
        w_h_init=initializers.Ones(),
        b_init=initializers.Ones())
    inputs = tf.random.uniform([self.batch_size, self.input_size])
    prev_state = core.initial_state(self.batch_size)
    core(inputs, prev_state)

    for v in core.variables:
      self.assertAllClose(self.evaluate(v), self.evaluate(tf.ones_like(v)))


def expit(x):
  return 1.0 / (1 + np.exp(-x))


class CuDNNGRUTest(test_utils.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()

    if self.primary_device != "GPU":
      self.skipTest("Only available on GPU")

    self.batch_size = 1
    self.input_size = 1
    self.hidden_size = 1

  @parameterized.parameters([1, 4])
  def testComputationAgainstTF(self, num_steps):
    inputs = tf.random.uniform([num_steps, self.batch_size, self.input_size])

    cudnn_gru = recurrent.CuDNNGRU(self.hidden_size)
    prev_state = cudnn_gru.initial_state(self.batch_size)
    outputs, states = cudnn_gru(inputs, prev_state)

    def cudnn_compatible_gru_fn(inputs, prev_state):
      # Sonnet `GRU` computes a_t and h_t as
      #
      #   a_t = tanh(W_{ia} x_t + W_{ha} (r_t h_{t-1}) + b_a)
      #   h_t = (1 - z_t) h_{t-1} + z_t a_t
      #
      # whereas CuDNN follows the original paper
      #
      #   a_t = tanh(W_{ia} x_t + r_t (W_{ha} h_{t-1}) + b_a)
      #   h_t = (1 - z_t) a_t + z_t h_{t-1}
      w_i = cudnn_gru.input_to_hidden
      w_h = cudnn_gru.hidden_to_hidden
      w_iz, w_ir, w_ia = tf.split(w_i, num_or_size_splits=3, axis=1)
      w_hz, w_hr, w_ha = tf.split(w_h, num_or_size_splits=3, axis=1)
      b_z, b_r, b_a = tf.split(cudnn_gru.b, num_or_size_splits=3)
      z = tf.sigmoid(
          tf.matmul(inputs, w_iz) + tf.matmul(prev_state, w_hz) + b_z)
      r = tf.sigmoid(
          tf.matmul(inputs, w_ir) + tf.matmul(prev_state, w_hr) + b_r)
      a = tf.tanh(
          tf.matmul(inputs, w_ia) + r * tf.matmul(prev_state, w_ha) + b_a)
      next_state = (1 - z) * a + z * prev_state
      return next_state, next_state

    expected_outputs, expected_final_state = recurrent.dynamic_unroll(
        cudnn_compatible_gru_fn, inputs, prev_state)

    self.assertAllClose(outputs, expected_outputs)
    self.assertAllClose(states[-1], expected_final_state)

  def testDtypeMismatch(self):
    core = recurrent.CuDNNGRU(hidden_size=self.hidden_size, dtype=tf.bfloat16)
    inputs = tf.random.uniform([1, self.batch_size, self.input_size])
    prev_state = core.initial_state(self.batch_size)
    self.assertIs(prev_state.dtype, tf.bfloat16)
    with self.assertRaisesRegex(
        TypeError, "inputs must have dtype tf.bfloat16, got tf.float32"):
      core(inputs, prev_state)

  def testInitialization(self):
    core = recurrent.CuDNNGRU(
        hidden_size=self.hidden_size,
        w_i_init=initializers.Ones(),
        w_h_init=initializers.Ones(),
        b_init=initializers.Ones())
    inputs = tf.random.uniform([1, self.batch_size, self.input_size])
    prev_state = core.initial_state(self.batch_size)
    core(inputs, prev_state)

    for v in core.variables:
      self.assertAllClose(self.evaluate(v), self.evaluate(tf.ones_like(v)))


class Counter(recurrent.RNNCore):
  """Count the steps.

  The output of the core at time step t is

      inputs * (h + t)

  where h is the hidden state which does not change with time.
  """

  def __init__(self, hidden_size, name=None):
    super().__init__(name)
    self._hidden_size = hidden_size
    self._built = False

  def __call__(self, inputs, prev_state):
    if not self._built:
      # Strictly speaking this variable is redundant, but all real-world
      # cores have variables, so Counter is no different.
      self.one = tf.Variable(1.0)
      self._built = True

    t, h = prev_state
    return inputs * (h + t), (t + self.one, h)

  def initial_state(self, batch_size):
    return (tf.cast(0.0, tf.float32), tf.zeros([batch_size, self._hidden_size]))


class Replicate(recurrent.RNNCore):
  """Replicate the output of the base RNN core."""

  def __init__(self, base_core, n, name=None):
    super().__init__(name)
    self._base_core = base_core
    self._n = n

  def __call__(self, inputs, prev_state):
    outputs, next_state = self._base_core(inputs, prev_state)
    return (outputs,) * self._n, next_state

  def initial_state(self, batch_size, **kwargs):
    return self._base_core.initial_state(batch_size, **kwargs)


class TrainableStateTest(test_utils.TestCase, parameterized.TestCase):

  @parameterized.parameters([
      {
          "initial_values_shape": []
      },
      {
          "initial_values_shape": tf.TensorShape([42])
      },
      {
          "initial_values_shape": (tf.TensorShape([4]), tf.TensorShape([2]))
      },
  ])
  def testUnmasked(self, initial_values_shape):
    trainable_state = recurrent.TrainableState(
        tree.map_structure(tf.ones, initial_values_shape))

    if initial_values_shape:
      self.assertEqual(
          len(trainable_state.trainable_variables), len(initial_values_shape))

    initial_state = trainable_state(batch_size=42)
    for s, shape in zip(
        tree.flatten(initial_state), tree.flatten(initial_values_shape)):
      self.assertEqual(s.shape, tf.TensorShape([42] + shape.as_list()))

  def testMasked(self):
    mask = (True, False)
    trainable_state = recurrent.TrainableState((tf.zeros([16]), tf.zeros([3])),
                                               mask)

    for var in trainable_state.trainable_variables:
      var.assign_add(tf.ones_like(var))

    initial_state = trainable_state(batch_size=42)
    for s, trainable in zip(tree.flatten(initial_state), tree.flatten(mask)):
      if trainable:
        self.assertNotAllClose(s, tf.zeros_like(s))
      else:
        self.assertAllClose(s, tf.zeros_like(s))

  def testForCore(self):
    core = recurrent.LSTM(hidden_size=16)
    trainable_state = recurrent.TrainableState.for_core(core)
    self.assertAllClose(
        trainable_state(batch_size=42), core.initial_state(batch_size=42))


@parameterized.parameters([
    {
        "use_tf_function": False,
        "unroll_fn": recurrent.dynamic_unroll
    },
    {
        "use_tf_function": False,
        "unroll_fn": recurrent.static_unroll
    },
    {
        "use_tf_function": True,
        "unroll_fn": recurrent.dynamic_unroll
    },
    {
        "use_tf_function": True,
        "unroll_fn": recurrent.static_unroll
    },
])
class UnrollTest(test_utils.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()

    self.num_steps = 5
    self.batch_size = 3
    self.hidden_size = 2
    self.core = Counter(self.hidden_size)

  def testFlat(self, use_tf_function, unroll_fn):
    if use_tf_function:
      unroll_fn = tf.function(unroll_fn)

    initial_state = _, h = self.core.initial_state(self.batch_size)
    input_sequence = tf.random.uniform([self.num_steps, self.batch_size, 1])
    output_sequence, final_state = unroll_fn(self.core, input_sequence,
                                             initial_state)

    self.assertAllClose(
        output_sequence,
        [inputs * (h + t) for t, inputs in enumerate(input_sequence)])
    self.assertAllClose(final_state, (tf.cast(self.num_steps, tf.float32), h))

  def testNestedInputs(self, use_tf_function, unroll_fn):
    if use_tf_function:
      unroll_fn = tf.function(unroll_fn)

    initial_state = _, h = self.core.initial_state(self.batch_size)
    input_sequence = tf.random.uniform([self.num_steps, self.batch_size, 1])
    output_sequence, final_state = unroll_fn(
        lambda inputs, prev_state: self.core(inputs["x"]["y"], prev_state),
        {"x": {
            "y": input_sequence
        }}, initial_state)

    self.assertAllClose(
        output_sequence,
        [inputs * (h + t) for t, inputs in enumerate(input_sequence)])
    self.assertAllClose(final_state, (tf.cast(self.num_steps, tf.float32), h))

  def testNestedOutputs(self, use_tf_function, unroll_fn):
    if use_tf_function:
      unroll_fn = tf.function(unroll_fn)

    num_replicas = 2
    core = Replicate(self.core, num_replicas)
    initial_state = _, h = core.initial_state(self.batch_size)
    input_sequence = tf.random.uniform([self.num_steps, self.batch_size, 1])
    output_sequence, final_state = unroll_fn(core, input_sequence,
                                             initial_state)

    expected_outputs = [
        inputs * (h + t) for t, inputs in enumerate(input_sequence)
    ]
    self.assertAllClose(output_sequence, (expected_outputs,) * num_replicas)
    self.assertAllClose(final_state, (tf.cast(self.num_steps, tf.float32), h))

  def testEmptyOutputs(self, use_tf_function, unroll_fn):
    if use_tf_function:
      unroll_fn = tf.function(unroll_fn)

    def core_fn(inputs, prev_state):
      return (inputs, tf.zeros(shape=(0,))), prev_state

    input_sequence = tf.random.uniform([self.num_steps, self.batch_size, 1])
    (_, empty), unused_final_state = unroll_fn(
        core_fn, input_sequence, initial_state=tf.constant(0.0))

    self.assertEqual(empty.shape, tf.TensorShape([self.num_steps, 0]))

  def testZeroSteps(self, use_tf_function, unroll_fn):
    if use_tf_function:
      unroll_fn = tf.function(unroll_fn)

    initial_state = self.core.initial_state(self.batch_size)
    input_sequence = tf.random.uniform([0, self.batch_size])

    with self.assertRaisesRegex(ValueError,
                                "must have at least a single time step"):
      unroll_fn(self.core, input_sequence, initial_state)

  def testInconsistentSteps(self, use_tf_function, unroll_fn):
    if use_tf_function:
      unroll_fn = tf.function(unroll_fn)

    initial_state = self.core.initial_state(self.batch_size)
    input_sequence = (tf.random.uniform([1, self.batch_size]),
                      tf.random.uniform([2, self.batch_size]))

    with self.assertRaisesRegex(ValueError,
                                "must have consistent number of time steps"):
      unroll_fn(self.core, input_sequence, initial_state)

  def testVariableLengthOneZeroLength(self, use_tf_function, unroll_fn):
    if use_tf_function:
      unroll_fn = tf.function(unroll_fn)

    sequence_length = tf.constant([0] + [self.num_steps] *
                                  (self.batch_size - 1))
    initial_state = self.core.initial_state(self.batch_size)
    input_sequence = tf.random.uniform([self.num_steps, self.batch_size, 1])
    output_sequence, _ = unroll_fn(
        self.core,
        input_sequence,
        initial_state,
        sequence_length=sequence_length)

    self.assertConsistentWithLength(output_sequence, sequence_length)

  def testVariableLengthRange(self, use_tf_function, unroll_fn):
    if use_tf_function:
      unroll_fn = tf.function(unroll_fn)

    sequence_length = tf.range(self.batch_size)
    initial_state = self.core.initial_state(self.batch_size)
    input_sequence = tf.random.uniform([self.num_steps, self.batch_size, 1])
    output_sequence, _ = unroll_fn(
        self.core,
        input_sequence,
        initial_state,
        sequence_length=sequence_length)

    self.assertConsistentWithLength(output_sequence, sequence_length)

  def assertConsistentWithLength(self, output_sequence, sequence_length):
    for t, _ in enumerate(output_sequence):
      for b in range(self.batch_size):
        if tf.equal(sequence_length[b], t):
          if t == 0:
            self.assertAllEqual(tf.reduce_sum(output_sequence[t, b]), 0.0)
          else:
            self.assertAllClose(output_sequence[t, b], output_sequence[t - 1,
                                                                       b])

  def testVariableLengthAllFull(self, use_tf_function, unroll_fn):
    if use_tf_function:
      unroll_fn = tf.function(unroll_fn)

    initial_state = self.core.initial_state(self.batch_size)
    input_sequence = tf.random.uniform([self.num_steps, self.batch_size, 1])
    output_sequence, final_state = unroll_fn(
        self.core,
        input_sequence,
        initial_state,
        sequence_length=tf.constant([self.num_steps] * self.batch_size))
    expected_output_sequence, expected_final_state = unroll_fn(
        self.core, input_sequence, initial_state)
    self.assertAllClose(output_sequence, expected_output_sequence)
    self.assertAllClose(final_state, expected_final_state)

  def testVariableLengthAllEmpty(self, use_tf_function, unroll_fn):
    if use_tf_function:
      unroll_fn = tf.function(unroll_fn)

    initial_state = self.core.initial_state(self.batch_size)
    input_sequence = tf.random.uniform([self.num_steps, self.batch_size, 1])
    output_sequence, final_state = unroll_fn(
        self.core,
        input_sequence,
        initial_state,
        sequence_length=tf.zeros(self.batch_size, tf.int32))
    self.assertAllClose(output_sequence, tf.zeros_like(output_sequence))
    # Scalars always get updates (to match `tf.nn.*_rnn` behavior).
    self.assertAllClose(final_state[0], self.num_steps)
    self.assertAllClose(final_state[1], initial_state[1])


class UnknownStepsUnrollTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()

    self.num_steps = 5
    self.batch_size = 3
    self.hidden_size = 2
    self.core = Counter(self.hidden_size)

  def testStaticUnroll(self):

    def do_unroll(input_sequence):
      initial_state = self.core.initial_state(self.batch_size)
      return recurrent.static_unroll(self.core, input_sequence, initial_state)

    with self.assertRaisesRegex(
        ValueError, "must have a statically known number of time steps"):
      tf.function(do_unroll).get_concrete_function(
          tf.TensorSpec([None, None, 1]))

  def testDynamicUnroll(self):

    def do_unroll(input_sequence):
      initial_state = self.core.initial_state(self.batch_size)
      return recurrent.dynamic_unroll(self.core, input_sequence, initial_state)

    cf = tf.function(do_unroll).get_concrete_function(
        tf.TensorSpec([None, None, 1]))
    output_sequence, unused_final_state = cf(
        tf.random.uniform([self.num_steps, self.batch_size, 1]))
    self.assertEqual(output_sequence.shape[0], self.num_steps)

  @unittest.skip("b/141910613")
  def testDynamicUnrollInconsistentSteps(self):

    def do_unroll(*input_sequence):
      return recurrent.dynamic_unroll(lambda inputs, _: inputs, input_sequence,
                                      ())

    cf = tf.function(do_unroll).get_concrete_function(
        tf.TensorSpec([None, None, 1]), tf.TensorSpec([None, None, 1]))
    with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                "must have consistent number of time steps"):
      cf(
          tf.random.uniform([self.num_steps, self.batch_size, 1]),
          tf.random.uniform([self.num_steps + 1, self.batch_size, 1]))


if __name__ == "__main__":
  tf.test.main()
