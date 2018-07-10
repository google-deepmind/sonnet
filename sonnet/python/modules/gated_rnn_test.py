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

"""Tests for recurrent cores in snt."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

# Dependency imports
from absl.testing import parameterized
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import sonnet as snt
import tensorflow as tf

from tensorflow.python.ops import variables


# Some helpers used for generic tests which cover both LSTM and BatchNormLSTM:


def _get_lstm_variable_names(lstm):
  if isinstance(lstm, snt.BatchNormLSTM):
    var_names = lstm.get_possible_initializer_keys(
        lstm.use_peepholes,
        lstm.use_batch_norm_h,
        lstm.use_batch_norm_x,
        lstm.use_batch_norm_c)
    if lstm.use_batch_norm_h or lstm.use_batch_norm_x:
      var_names |= {"w_x", "w_h"}
    else:
      var_names |= {"w_xh"}
  else:
    var_names = lstm.get_possible_initializer_keys(lstm.use_peepholes)
    var_names |= {"w_xh"}
  var_names -= {"w_gates", "b_gates"}
  var_names |= {"b"}
  return var_names


def _construct_lstm(use_batch_norm_h=False, use_batch_norm_x=False,
                    use_batch_norm_c=False, max_unique_stats=1, **kwargs):
  if any([use_batch_norm_h, use_batch_norm_x, use_batch_norm_c]):
    cell = snt.BatchNormLSTM(
        use_batch_norm_h=use_batch_norm_h,
        use_batch_norm_x=use_batch_norm_x,
        use_batch_norm_c=use_batch_norm_c,
        max_unique_stats=max_unique_stats,
        **kwargs)
    return cell, cell.with_batch_norm_control(is_training=True)
  else:
    cell = snt.LSTM(**kwargs)
    return cell, cell


def _get_possible_initializer_keys(use_peepholes, use_batch_norm_h,
                                   use_batch_norm_x, use_batch_norm_c):
  if any([use_batch_norm_h, use_batch_norm_x, use_batch_norm_c]):
    return snt.BatchNormLSTM.get_possible_initializer_keys(
        use_peepholes, use_batch_norm_h, use_batch_norm_x, use_batch_norm_c)
  else:
    return snt.LSTM.get_possible_initializer_keys(use_peepholes)


# @tf.contrib.eager.run_all_tests_in_graph_and_eager_modes
class LSTMTest(tf.test.TestCase, parameterized.TestCase):

  def testShape(self):
    batch_size = 2
    hidden_size = 4
    inputs = tf.ones(dtype=tf.float32, shape=[batch_size, hidden_size])
    prev_hidden = tf.ones(dtype=tf.float32, shape=[batch_size, hidden_size])
    prev_cell = tf.ones(dtype=tf.float32, shape=[batch_size, hidden_size])
    lstm = snt.LSTM(hidden_size)
    output, next_state = lstm(inputs, (prev_hidden, prev_cell))

    shape = np.ndarray((batch_size, hidden_size))

    self.assertShapeEqual(shape, next_state[0])
    self.assertShapeEqual(shape, next_state[1])
    self.assertShapeEqual(shape, output)

  def testVariables(self):
    batch_size = 5
    hidden_size = 20
    mod_name = "rnn"
    inputs = tf.ones(dtype=tf.float32, shape=[batch_size, hidden_size])
    prev_cell = tf.ones(dtype=tf.float32, shape=[batch_size, hidden_size])
    prev_hidden = tf.ones(dtype=tf.float32, shape=[batch_size, hidden_size])
    lstm = snt.LSTM(hidden_size, name=mod_name)
    self.assertEqual(lstm.scope_name, mod_name)
    with self.assertRaisesRegexp(snt.Error, "not instantiated yet"):
      lstm.get_variables()
    lstm(inputs, (prev_hidden, prev_cell))

    lstm_variables = lstm.get_variables()
    self.assertEqual(len(lstm_variables), 2, "LSTM should have 2 variables")
    param_map = {param.name.split("/")[-1].split(":")[0]:
                 param for param in lstm_variables}

    self.assertShapeEqual(
        np.ndarray(4 * hidden_size),
        tf.convert_to_tensor(param_map[snt.LSTM.B_GATES]))
    self.assertShapeEqual(
        np.ndarray((2 * hidden_size, 4 * hidden_size)),
        tf.convert_to_tensor(param_map[snt.LSTM.W_GATES]))

  @parameterized.named_parameters(
      [("lstm", None), ("lstm_with_recurrent_projection", 6)])
  def testComputation(self, projection_size):
    batch_size = 2
    hidden_size = 4
    hidden_state_size = projection_size or hidden_size

    # With random data, check the TF calculation matches the Numpy version.
    input_data = np.random.randn(batch_size, hidden_size).astype(np.float32)
    prev_hidden_data = np.random.randn(batch_size,
                                       hidden_state_size).astype(np.float32)
    prev_cell_data = np.random.randn(batch_size, hidden_size).astype(np.float32)

    inputs = tf.constant(input_data)
    prev_cell = tf.constant(prev_cell_data)
    prev_hidden = tf.constant(prev_hidden_data)

    lstm = snt.LSTM(hidden_size, projection_size=projection_size)
    _, next_state = lstm(inputs, (prev_hidden, prev_cell))
    next_hidden, next_cell = next_state
    lstm_variables = lstm.get_variables()
    param_map = {param.name.split("/")[-1].split(":")[0]:
                 param for param in lstm_variables}

    self.evaluate(tf.global_variables_initializer())
    fetches = [(next_hidden, next_cell), param_map[snt.LSTM.W_GATES],
               param_map[snt.LSTM.B_GATES]]
    if projection_size is not None:
      fetches.append(param_map[snt.LSTM.W_H_PROJECTION])
    output = self.evaluate(fetches)

    next_state_ex, gate_weights_ex, gate_biases_ex = output[:3]
    in_and_hid = np.concatenate((input_data, prev_hidden_data), axis=1)
    real_gate = np.dot(in_and_hid, gate_weights_ex) + gate_biases_ex
    # i = input_gate, j = next_input, f = forget_gate, o = output_gate
    i, j, f, o = np.hsplit(real_gate, 4)
    real_cell = (prev_cell_data / (1 + np.exp(-(f + lstm._forget_bias))) +
                 1 / (1 + np.exp(-i)) * np.tanh(j))
    real_hidden = np.tanh(real_cell) * 1 / (1 + np.exp(-o))
    if projection_size is not None:
      real_hidden = np.matmul(real_hidden, output[-1])

    self.assertAllClose(real_hidden, next_state_ex[0])
    self.assertAllClose(real_cell, next_state_ex[1])

  def testPeephole(self):
    batch_size = 5
    hidden_size = 20

    # With random data, check the TF calculation matches the Numpy version.
    input_data = np.random.randn(batch_size, hidden_size).astype(np.float32)
    prev_hidden_data = np.random.randn(batch_size,
                                       hidden_size).astype(np.float32)
    prev_cell_data = np.random.randn(batch_size, hidden_size).astype(np.float32)

    # Initialize the rnn and verify the number of parameter sets.
    inputs = tf.constant(input_data)
    prev_cell = tf.constant(prev_cell_data)
    prev_hidden = tf.constant(prev_hidden_data)

    lstm = snt.LSTM(hidden_size, use_peepholes=True)
    _, next_state = lstm(inputs, (prev_hidden, prev_cell))
    next_hidden, next_cell = next_state
    lstm_variables = lstm.get_variables()
    self.assertEqual(len(lstm_variables), 5, "LSTM should have 5 variables")

    # Unpack parameters into dict and check their sizes.
    param_map = {param.name.split("/")[-1].split(":")[0]:
                 param for param in lstm_variables}
    self.assertShapeEqual(
        np.ndarray(4 * hidden_size),
        tf.convert_to_tensor(param_map[snt.LSTM.B_GATES]))
    self.assertShapeEqual(
        np.ndarray((2 * hidden_size, 4 * hidden_size)),
        tf.convert_to_tensor(param_map[snt.LSTM.W_GATES]))
    self.assertShapeEqual(
        np.ndarray(hidden_size),
        tf.convert_to_tensor(param_map[snt.LSTM.W_F_DIAG]))
    self.assertShapeEqual(
        np.ndarray(hidden_size),
        tf.convert_to_tensor(param_map[snt.LSTM.W_I_DIAG]))
    self.assertShapeEqual(
        np.ndarray(hidden_size),
        tf.convert_to_tensor(param_map[snt.LSTM.W_O_DIAG]))

    self.evaluate(tf.global_variables_initializer())
    fetches = [(next_hidden, next_cell), param_map[snt.LSTM.W_GATES],
               param_map[snt.LSTM.B_GATES], param_map[snt.LSTM.W_F_DIAG],
               param_map[snt.LSTM.W_I_DIAG], param_map[snt.LSTM.W_O_DIAG]]
    output = self.evaluate(fetches)

    next_state_ex, w_ex, b_ex, wfd_ex, wid_ex, wod_ex = output
    in_and_hid = np.concatenate((input_data, prev_hidden_data), axis=1)
    real_gate = np.dot(in_and_hid, w_ex) + b_ex
    # i = input_gate, j = next_input, f = forget_gate, o = output_gate
    i, j, f, o = np.hsplit(real_gate, 4)
    real_cell = (prev_cell_data /
                 (1 + np.exp(-(f + lstm._forget_bias +
                               wfd_ex * prev_cell_data))) +
                 1 / (1 + np.exp(-(i + wid_ex * prev_cell_data))) * np.tanh(j))
    real_hidden = (np.tanh(real_cell + wod_ex * real_cell) *
                   1 / (1 + np.exp(-o)))

    self.assertAllClose(real_hidden, next_state_ex[0])
    self.assertAllClose(real_cell, next_state_ex[1])

  @parameterized.parameters(
      *itertools.product(
          (True, False), (True, False), (True, False), (True, False))
  )
  def testInitializers(self, use_peepholes, use_batch_norm_h, use_batch_norm_x,
                       use_batch_norm_c):
    batch_size = 2
    hidden_size = 4

    keys = _get_possible_initializer_keys(
        use_peepholes, use_batch_norm_h, use_batch_norm_x, use_batch_norm_c)
    initializers = {
        key: tf.constant_initializer(1.5) for key in keys
    }

    # Test we can successfully create the LSTM with initializers.
    lstm, wrapped_lstm = _construct_lstm(hidden_size=hidden_size,
                                         use_peepholes=use_peepholes,
                                         use_batch_norm_h=use_batch_norm_h,
                                         use_batch_norm_x=use_batch_norm_x,
                                         use_batch_norm_c=use_batch_norm_c,
                                         initializers=initializers)

    # Test we can build the LSTM.
    inputs = tf.ones(dtype=tf.float32, shape=[batch_size, hidden_size])
    prev_cell = tf.ones(dtype=tf.float32, shape=[batch_size, hidden_size])
    prev_hidden = tf.ones(dtype=tf.float32, shape=[batch_size, hidden_size])
    wrapped_lstm(inputs, (prev_hidden, prev_cell))
    init = tf.global_variables_initializer()

    # Test that the initializers have been correctly applied.
    lstm_variable_names = _get_lstm_variable_names(lstm)
    lstm_variables = [getattr(lstm, "_" + name) for name in lstm_variable_names]

    self.evaluate(init)
    lstm_variables_v = self.evaluate(lstm_variables)
    for lstm_variable_v in lstm_variables_v:
      self.assertAllClose(lstm_variable_v, 1.5 * np.ones(lstm_variable_v.shape))

  def testPeepholeInitializersCheck(self):
    hidden_size = 4

    # Test that passing in a peephole initializer when we don't request peephole
    # connections raises an error.
    for key in [snt.LSTM.W_F_DIAG, snt.LSTM.W_I_DIAG, snt.LSTM.W_O_DIAG]:
      with self.assertRaisesRegexp(KeyError, "Invalid initializer"):
        snt.LSTM(hidden_size, use_peepholes=False,
                 initializers={key: tf.constant_initializer(0)})

  @parameterized.parameters(
      (1e-8, 14),
      (0.5, None),
      (1 - 1e-8, 0)
  )
  def testRecurrentDropout(self, keep_prob, expected_zeros):
    """Performs various recurrent dropout checks.

    - The training and testing versions have the same output when the keep prob
      is very close to 1.
    - The return is deterministic for keep probs 0 or close to 1.
    - The final hidden state has 0s at the same position as the mask.

    Args:
      keep_prob: the recurrent dropout keep probability.
      expected_zeros: the number of expected zeros in the dropout mask.
    """

    batch_size = 2
    input_size = 3
    hidden_size = 7
    seq_len = 5

    train_cell, test_cell = snt.lstm_with_recurrent_dropout(
        hidden_size, keep_prob=keep_prob)
    input_data = np.stack(
        [np.random.rand(seq_len, input_size)] * batch_size).astype(np.float32)
    inputs = tf.constant(input_data)
    train_output, ((train_hidden, _), [train_mask]) = tf.nn.dynamic_rnn(
        train_cell,
        inputs,
        initial_state=train_cell.initial_state(batch_size, tf.float32),
        dtype=tf.float32)
    valid_output, _ = tf.nn.dynamic_rnn(
        test_cell,
        inputs,
        initial_state=test_cell.initial_state(
            batch_size, tf.float32),
        dtype=tf.float32)

    self.evaluate(tf.global_variables_initializer())
    # Use the same input data for each row.
    input_data = np.stack([np.random.rand(seq_len, input_size)] * batch_size)

    train_out, valid_out, hidden, mask = self.evaluate(
        [train_output, valid_output, train_hidden, train_mask])
    almost_one = abs(1 - keep_prob) < 1e-5
    if almost_one:
      self.assertAllClose(train_out, valid_out)
    else:
      self.assertGreater(np.max(train_out - valid_out), 0.001)

    self.assertAllClose(valid_out[0], valid_out[1])
    deterministic = almost_one or abs(keep_prob < 1e-5)
    if deterministic:
      self.assertAllClose(train_out[0], train_out[1])
      self.assertEqual(expected_zeros, np.sum(hidden == 0))
      self.assertEqual(expected_zeros, np.sum(mask == 0))
    else:
      self.assertGreater(np.max(train_out[0] - train_out[1]), 0.001)

    self.assertAllEqual(mask == 0, hidden == 0)

  @parameterized.parameters(
      (1 - 1e-8, 0, 0),
      (0.5, None, None),
      (1e-8, 14, 14)
  )
  def testZoneout(self, keep_prob, expected_frozen_h, expected_frozen_c):
    """Performs various zoneout checks.

    The training and testing versions have the same output when keep_prob
    is close to 0 or close to 1. The returned output for the training version
    is also deterministic in this case.

    Args:
      keep_prob: the probability to use the updated version of the state.
      expected_frozen_h: the number of hidden state values that are left
        unchanged after applying one step of LSTM with zoneout.
      expected_frozen_c: the number of cell state values that are left
        unchanged after applying one step of LSTM with zoneout.
    """

    batch_size = 2
    input_size = 3
    hidden_size = 7
    seq_len = 5

    train_cell, test_cell = snt.lstm_with_zoneout(
        hidden_size, keep_prob_c=keep_prob, keep_prob_h=keep_prob)
    # Use the same input data for each row.
    input_data = np.stack(
        [np.random.rand(seq_len, input_size)] * batch_size).astype(np.float32)
    inputs = tf.constant(input_data)
    train_output, (train_h, train_c) = tf.nn.dynamic_rnn(
        train_cell,
        inputs,
        dtype=tf.float32)
    _, (next_train_h, next_train_c) = train_cell(
        inputs[:, 0], snt.LSTMState(train_h, train_c))
    valid_output, _ = tf.nn.dynamic_rnn(
        test_cell,
        inputs,
        dtype=tf.float32)

    self.evaluate(tf.global_variables_initializer())

    outputs = self.evaluate({
        "train_out": train_output,
        "valid_out": valid_output,
        "train_h": train_h,
        "train_c": train_c,
        "next_train_h": next_train_h,
        "next_train_c": next_train_c,
    })

    self.assertAllClose(outputs["valid_out"][0], outputs["valid_out"][1])
    deterministic = abs(1 - keep_prob) < 1e-5 or abs(keep_prob < 1e-5)
    if deterministic:
      self.assertAllClose(outputs["train_out"], outputs["valid_out"])
      self.assertAllClose(outputs["train_out"][0], outputs["train_out"][1])
      self.assertEqual(expected_frozen_h,
                       np.sum(outputs["train_h"] == outputs["next_train_h"]))
      self.assertEqual(expected_frozen_c,
                       np.sum(outputs["train_c"] == outputs["next_train_c"]))
    else:
      # Ensure that the training and validation outputs are different.
      self.assertGreater(
          np.max(np.abs(outputs["train_out"] - outputs["valid_out"])), 0.05)
      # Ensure that the training output is not deterministic.
      self.assertGreater(
          np.max(np.abs(outputs["train_out"][0] - outputs["train_out"][1])),
          0.04)

  @parameterized.parameters(
      (True, False, False),
      (False, True, False),
      (False, False, True)
  )
  def testBatchNormBuildFlag(self, use_batch_norm_h, use_batch_norm_x,
                             use_batch_norm_c):
    """Check if an error is raised if we don't specify the is_training flag."""
    batch_size = 2
    hidden_size = 4

    inputs = tf.ones(dtype=tf.float32, shape=[batch_size, hidden_size])
    prev_cell = tf.ones(dtype=tf.float32, shape=[batch_size, hidden_size])
    prev_hidden = tf.ones(dtype=tf.float32, shape=[batch_size, hidden_size])

    err = "is_training flag must be explicitly specified"
    with self.assertRaisesRegexp(ValueError, err):
      lstm = snt.BatchNormLSTM(
          hidden_size,
          use_batch_norm_h=use_batch_norm_h,
          use_batch_norm_x=use_batch_norm_x,
          use_batch_norm_c=use_batch_norm_c)
      lstm(inputs, (prev_cell, prev_hidden))

  def testBatchNormInitializersCheck(self):
    hidden_size = 4

    # Test that passing in a batchnorm initializer when we don't request
    # that form of batchnorm raises an error.
    for key, options in [
        (snt.BatchNormLSTM.GAMMA_H, {"use_batch_norm_h": False,
                                     "use_batch_norm_x": True}),
        (snt.BatchNormLSTM.GAMMA_X, {"use_batch_norm_x": False,
                                     "use_batch_norm_h": True}),
        (snt.BatchNormLSTM.GAMMA_C, {"use_batch_norm_c": False,
                                     "use_batch_norm_h": True}),
        (snt.BatchNormLSTM.BETA_C, {"use_batch_norm_c": False,
                                    "use_batch_norm_h": True})]:
      with self.assertRaisesRegexp(KeyError, "Invalid initializer"):
        snt.BatchNormLSTM(
            hidden_size, initializers={key: tf.constant_initializer(0)},
            **options)

  @parameterized.parameters(
      *itertools.product(
          (True, False), (True, False), (True, False), (True, False))
  )
  def testPartitioners(self, use_peepholes, use_batch_norm_h, use_batch_norm_x,
                       use_batch_norm_c):
    if tf.executing_eagerly():
      self.skipTest("Partitioned variables arenot supported in eager mode.")
    batch_size = 2
    hidden_size = 4

    keys = _get_possible_initializer_keys(
        use_peepholes, use_batch_norm_h, use_batch_norm_x, use_batch_norm_c)
    partitioners = {
        key: tf.variable_axis_size_partitioner(10) for key in keys
    }

    # Test we can successfully create the LSTM with partitioners.
    lstm, wrapped_lstm = _construct_lstm(hidden_size=hidden_size,
                                         use_peepholes=use_peepholes,
                                         use_batch_norm_h=use_batch_norm_h,
                                         use_batch_norm_x=use_batch_norm_x,
                                         use_batch_norm_c=use_batch_norm_c,
                                         partitioners=partitioners)

    # Test we can build the LSTM
    inputs = tf.ones(dtype=tf.float32, shape=[batch_size, hidden_size])
    prev_cell = tf.ones(dtype=tf.float32, shape=[batch_size, hidden_size])
    prev_hidden = tf.ones(dtype=tf.float32, shape=[batch_size, hidden_size])
    wrapped_lstm(inputs, (prev_hidden, prev_cell))

    # Test that the variables are partitioned.
    var_names = _get_lstm_variable_names(lstm)
    for var_name in var_names:
      self.assertEqual(type(getattr(lstm, "_" + var_name)),
                       variables.PartitionedVariable)

  @parameterized.parameters(
      *itertools.product(
          (True, False), (True, False), (True, False), (True, False))
  )
  def testRegularizers(self, use_peepholes, use_batch_norm_h, use_batch_norm_x,
                       use_batch_norm_c):
    batch_size = 2
    hidden_size = 4

    keys = _get_possible_initializer_keys(
        use_peepholes, use_batch_norm_h, use_batch_norm_x, use_batch_norm_c)
    regularizers = {
        key: tf.nn.l2_loss for key in keys
    }

    # Test we can successfully create the LSTM with regularizers.
    _, wrapped_lstm = _construct_lstm(hidden_size=hidden_size,
                                      use_peepholes=use_peepholes,
                                      use_batch_norm_h=use_batch_norm_h,
                                      use_batch_norm_x=use_batch_norm_x,
                                      use_batch_norm_c=use_batch_norm_c,
                                      regularizers=regularizers)

    # Test we can build the LSTM
    inputs = tf.ones(dtype=tf.float32, shape=[batch_size, hidden_size])
    prev_cell = tf.ones(dtype=tf.float32, shape=[batch_size, hidden_size])
    prev_hidden = tf.ones(dtype=tf.float32, shape=[batch_size, hidden_size])
    wrapped_lstm(inputs, (prev_hidden, prev_cell))

    # Test that we have regularization losses.
    num_reg_losses = len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    if use_batch_norm_h or use_batch_norm_x:
      self.assertEqual(num_reg_losses, len(keys) + 1)
    else:
      self.assertEqual(num_reg_losses, len(keys))

  # Pick some hopefully representative combination of parameter values
  # (want to test with seq_len < max_unique_stats and seq_len >
  # max_unique_stats, and some other combinations for good measure).
  @parameterized.parameters(
      (False, 1, 1, 2),
      (True, 3, 1, 2),
      (False, 1, 2, 1),
      (True, 3, 2, 1),
      (False, 3, 3, 5))
  def testSameInStaticAndDynamicWithBatchNorm(self, use_peepholes, batch_size,
                                              max_unique_stats, seq_len):
    # Tests that when the cell is used in either a normal tensorflow rnn, or in
    # tensorflow's dynamic_rnn, that the output is the same. This is to test
    # test that the cores aren't doing anything funny they shouldn't be (like
    # relying on the number of times they've been invoked).

    hidden_size = 3
    input_size = 3

    inputs = tf.ones(
        dtype=tf.float32,
        shape=[batch_size, seq_len, input_size],
        name="inputs")
    static_inputs = tf.unstack(inputs, axis=1)

    test_local_stats = False

    cell = snt.BatchNormLSTM(
        hidden_size=hidden_size,
        max_unique_stats=max_unique_stats,
        use_peepholes=use_peepholes,
        use_batch_norm_h=True,
        use_batch_norm_x=True,
        use_batch_norm_c=True)

    def connect(training):
      static_output_unpacked, _ = tf.contrib.rnn.static_rnn(
          cell.with_batch_norm_control(is_training=training,
                                       test_local_stats=test_local_stats),
          static_inputs,
          initial_state=cell.initial_state(batch_size, tf.float32))

      static_output = tf.stack(static_output_unpacked, axis=1)

      dynamic_output, _ = tf.nn.dynamic_rnn(
          cell.with_batch_norm_control(is_training=training,
                                       test_local_stats=test_local_stats),
          inputs,
          initial_state=cell.initial_state(batch_size, tf.float32),
          dtype=tf.float32)

      return static_output, dynamic_output

    ops = connect(training=True)
    self.evaluate(tf.global_variables_initializer())
    static_out, dynamic_out = self.evaluate(ops)
    self.assertAllClose(static_out, dynamic_out)

    # Do a pass to train the exponential moving statistics.
    for _ in range(5):
      if tf.executing_eagerly():
        ops = connect(training=True)
      static_out, dynamic_out = self.evaluate(ops)
      self.assertAllClose(static_out, dynamic_out)

    # And check that same when using test statistics.
    ops = connect(training=False)
    static_out, dynamic_out = self.evaluate(ops)
    self.assertAllClose(static_out, dynamic_out)

  def testSameInStaticAndDynamic(self):
    batch_size = 3
    seq_len = 2
    hidden_size = 3
    input_size = 3

    inputs = tf.ones(
        dtype=tf.float32,
        shape=[batch_size, seq_len, input_size],
        name="inputs")
    static_inputs = tf.unstack(inputs, axis=1)

    cell = snt.LSTM(hidden_size=hidden_size)

    static_output_unpacked, _ = tf.contrib.rnn.static_rnn(
        cell, static_inputs,
        initial_state=cell.initial_state(batch_size, tf.float32))

    dynamic_output, _ = tf.nn.dynamic_rnn(
        cell, inputs,
        initial_state=cell.initial_state(batch_size, tf.float32),
        dtype=tf.float32)

    static_output = tf.stack(static_output_unpacked, axis=1)

    self.evaluate(tf.global_variables_initializer())

    # Check that static and dynamic give the same output
    static_out, dynamic_out = self.evaluate([static_output, dynamic_output])
    self.assertAllClose(static_out, dynamic_out)

  def testLayerNormVariables(self):
    core = snt.LSTM(hidden_size=3, use_layer_norm=True)

    batch_size = 3
    inputs = tf.ones(dtype=tf.float32, shape=[batch_size, 3, 3])
    tf.nn.dynamic_rnn(core,
                      inputs,
                      initial_state=core.initial_state(batch_size, tf.float32))

    self.assertTrue(core.use_layer_norm)

    expected = 4  # gate bias and one weight, plus LayerNorm's gamma, beta.
    self.assertEqual(len(core.get_variables()), expected)

  def testHiddenClipping(self):
    core = snt.LSTM(hidden_size=5, hidden_clip_value=1.0)
    obs = tf.constant(np.random.rand(3, 10), dtype=tf.float32)

    unclipped = np.random.rand(3, 5) - 0.5
    unclipped *= 2.0 / unclipped.max()
    unclipped = unclipped.astype(np.float32)
    clipped = unclipped.clip(-1., 1.)

    hidden = tf.constant(unclipped)
    cell = tf.constant(unclipped)
    output = core(obs, [hidden, cell])
    self.evaluate(tf.global_variables_initializer())
    output1, (hidden1, cell1) = self.evaluate(output)

    hidden = tf.constant(clipped)
    output = core(obs, [hidden, cell])

    output2, (hidden2, cell2) = self.evaluate(output)
    self.assertAllClose(output1, output2)
    self.assertAllClose(hidden1, hidden2)
    self.assertAllClose(cell1, cell2)

  def testCellClipping(self):
    core = snt.LSTM(hidden_size=5, cell_clip_value=1.0)
    obs = tf.constant(np.random.rand(3, 10), dtype=tf.float32)

    unclipped = np.random.rand(3, 5) - 0.5
    unclipped *= 2.0 / unclipped.max()
    unclipped = unclipped.astype(np.float32)
    clipped = unclipped.clip(-1., 1.)

    hidden = tf.constant(unclipped)
    cell = tf.constant(unclipped)
    output = core(obs, [hidden, cell])
    self.evaluate(tf.global_variables_initializer())
    output1, (hidden1, cell1) = self.evaluate(output)

    cell = tf.constant(clipped)
    output = core(obs, [hidden, cell])
    output2, (hidden2, cell2) = self.evaluate(output)
    self.assertAllClose(output1, output2)
    self.assertAllClose(hidden1, hidden2)
    self.assertAllClose(cell1, cell2)

  @parameterized.parameters(
      (False, False, False, False),
      (False, True, False, False),
      (True, False, True, False),
      (False, True, True, False),
      (False, False, False, True),
      (True, True, False, True),
      (False, False, True, True),
      (False, True, True, True))
  def testBatchNormVariables(self,
                             use_peepholes,
                             use_batch_norm_h,
                             use_batch_norm_x,
                             use_batch_norm_c):
    cell, wrapped_cell = _construct_lstm(hidden_size=3,
                                         use_peepholes=use_peepholes,
                                         use_batch_norm_h=use_batch_norm_h,
                                         use_batch_norm_x=use_batch_norm_x,
                                         use_batch_norm_c=use_batch_norm_c)

    # Need to connect the cell before it has variables
    batch_size = 3
    inputs = tf.ones(dtype=tf.float32, shape=[batch_size, 3, 3])
    tf.nn.dynamic_rnn(wrapped_cell, inputs,
                      initial_state=cell.initial_state(batch_size, tf.float32))

    self.assertEqual(use_peepholes, cell.use_peepholes)
    if use_batch_norm_h or use_batch_norm_x or use_batch_norm_c:
      self.assertEqual(use_batch_norm_h, cell.use_batch_norm_h)
      self.assertEqual(use_batch_norm_x, cell.use_batch_norm_x)
      self.assertEqual(use_batch_norm_c, cell.use_batch_norm_c)

    if use_batch_norm_h or use_batch_norm_x:
      expected = 3  # gate bias and two weights
    else:
      expected = 2  # gate bias and weight
    if use_peepholes:
      expected += 3
    if use_batch_norm_h:
      expected += 1  # gamma_h
    if use_batch_norm_x:
      expected += 1  # gamma_x
    if use_batch_norm_c:
      expected += 2  # gamma_c, beta_c

    self.assertEqual(len(cell.get_variables()), expected)

  def testCheckMaxUniqueStats(self):
    self.assertRaisesRegexp(ValueError,
                            ".*must be >= 1",
                            snt.BatchNormLSTM,
                            hidden_size=1,
                            max_unique_stats=0)

  @parameterized.parameters(
      (False, 1),
      (False, 2),
      (True, 1),
      (True, 2))
  def testTraining(self, trainable_initial_state, max_unique_stats):
    """Test that everything trains OK, with or without trainable init. state."""
    hidden_size = 3
    batch_size = 3
    time_steps = 3
    cell = snt.BatchNormLSTM(hidden_size=hidden_size,
                             max_unique_stats=max_unique_stats)
    inputs = tf.constant(np.random.rand(batch_size, time_steps, 3),
                         dtype=tf.float32)
    initial_state = cell.initial_state(
        batch_size, tf.float32, trainable_initial_state)
    def loss_fn():
      output, _ = tf.nn.dynamic_rnn(
          cell.with_batch_norm_control(is_training=True),
          inputs,
          initial_state=initial_state,
          dtype=tf.float32)

      return tf.reduce_mean(tf.square(
          output - np.random.rand(batch_size, time_steps, hidden_size)))

    train_op = tf.train.GradientDescentOptimizer(1).minimize(
        loss_fn if tf.executing_eagerly() else loss_fn())

    init = tf.global_variables_initializer()
    self.evaluate(init)
    self.evaluate(train_op)

  # Regression test.
  def testSideBySide(self):
    hidden_size = 3
    batch_size = 4
    lstm1 = snt.LSTM(hidden_size=hidden_size)
    lstm2 = snt.LSTM(hidden_size=hidden_size)
    lstm1.initial_state(batch_size, trainable=True)
    # Previously either of the two lines below would cause a crash due to
    # Variable name collision.
    lstm1.initial_state(batch_size, trainable=True)
    lstm2.initial_state(batch_size, trainable=True)

  def testInitialStateNames(self):
    if tf.executing_eagerly():
      self.skipTest("Tensor name is not supported in eager mode.")
    lstm = snt.LSTM(hidden_size=3, name="foo")
    unnamed_init_state = lstm.initial_state(4, trainable=True)
    named_init_state = lstm.initial_state(4, trainable=True, name="bar")
    self.assertEqual(unnamed_init_state[0].name,
                     "foo_initial_state/state_hidden_tiled:0")
    self.assertEqual(unnamed_init_state[1].name,
                     "foo_initial_state/state_cell_tiled:0")
    self.assertEqual(named_init_state[0].name, "bar/state_hidden_tiled:0")
    self.assertEqual(named_init_state[1].name, "bar/state_cell_tiled:0")


# @tf.contrib.eager.run_all_tests_in_graph_and_eager_modes
class ConvLSTMTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      (snt.Conv1DLSTM, 1, False),
      (snt.Conv1DLSTM, 1, True),
      (snt.Conv2DLSTM, 2, False),
      (snt.Conv2DLSTM, 2, True),
  )
  def testShape(self, lstm_class, dim, use_bias):
    batch_size = 2
    input_shape = (8,) * dim
    input_channels = 3
    output_channels = 5

    input_shape = (batch_size,) + input_shape + (input_channels,)
    output_shape = input_shape[:-1] + (output_channels,)

    inputs = tf.ones(dtype=tf.float32, shape=input_shape)
    prev_hidden = tf.ones(dtype=tf.float32, shape=output_shape)
    prev_cell = tf.ones(dtype=tf.float32, shape=output_shape)
    lstm = lstm_class(
        input_shape=input_shape[1:],
        output_channels=output_channels,
        kernel_shape=1,
        use_bias=use_bias)
    output, next_state = lstm(inputs, (prev_hidden, prev_cell))

    expected_shape = np.ndarray(output_shape)

    self.assertShapeEqual(expected_shape, next_state[0])
    self.assertShapeEqual(expected_shape, next_state[1])
    self.assertShapeEqual(expected_shape, output)

  @parameterized.parameters(
      (snt.Conv1DLSTM, 1, False, False),
      (snt.Conv1DLSTM, 1, True, False),
      (snt.Conv2DLSTM, 2, False, False),
      (snt.Conv2DLSTM, 2, True, False),
      (snt.Conv1DLSTM, 1, False, True),
      (snt.Conv1DLSTM, 1, True, True),
      (snt.Conv2DLSTM, 2, False, True),
      (snt.Conv2DLSTM, 2, True, True),
  )
  def testInitializers(self, lstm_class, dim, use_bias, legacy_bias_behaviour):
    keys = snt.Conv2DLSTM.get_possible_initializer_keys(use_bias)
    initializers = {
        key: tf.constant_initializer(i) for i, key in enumerate(keys)
    }

    batch_size = 2
    input_shape = (8,) * dim
    input_channels = 3
    output_channels = 5

    input_shape = (batch_size,) + input_shape + (input_channels,)
    output_shape = input_shape[:-1] + (output_channels,)

    inputs = tf.ones(dtype=tf.float32, shape=input_shape)
    prev_hidden = tf.ones(dtype=tf.float32, shape=output_shape)
    prev_cell = tf.ones(dtype=tf.float32, shape=output_shape)

    # Test we can successfully create the LSTM with partitioners.
    lstm = lstm_class(
        input_shape=input_shape[1:],
        output_channels=output_channels,
        kernel_shape=1,
        use_bias=use_bias,
        legacy_bias_behaviour=legacy_bias_behaviour,
        initializers=initializers)
    lstm(inputs, (prev_hidden, prev_cell))

    init = tf.global_variables_initializer()

    # Test that the initializers have been applied correctly.
    self.evaluate(init)
    for conv_key, convolution in lstm.convolutions.items():
      for i, key in enumerate(keys):
        if not legacy_bias_behaviour and conv_key == "hidden" and key == "b":
          self.assertFalse(hasattr(convolution, key))
          continue
        variable = getattr(convolution, key)
        self.assertAllClose(self.evaluate(variable),
                            np.full(variable.get_shape(),
                                    i, dtype=np.float32))

  @parameterized.parameters(
      (snt.Conv1DLSTM, 1, False, False),
      (snt.Conv1DLSTM, 1, True, False),
      (snt.Conv2DLSTM, 2, False, False),
      (snt.Conv2DLSTM, 2, True, False),
      (snt.Conv1DLSTM, 1, False, True),
      (snt.Conv1DLSTM, 1, True, True),
      (snt.Conv2DLSTM, 2, False, True),
      (snt.Conv2DLSTM, 2, True, True),
  )
  def testPartitioners(self, lstm_class, dim, use_bias, legacy_bias_behaviour):
    if tf.executing_eagerly():
      self.skipTest("Partitioned variables are not supported in eager.")
    keys = snt.Conv2DLSTM.get_possible_initializer_keys(use_bias)
    partitioners = {
        key: tf.variable_axis_size_partitioner(10) for key in keys
    }

    batch_size = 2
    input_shape = (8,) * dim
    input_channels = 3
    output_channels = 5

    input_shape = (batch_size,) + input_shape + (input_channels,)
    output_shape = input_shape[:-1] + (output_channels,)

    inputs = tf.ones(dtype=tf.float32, shape=input_shape)
    prev_hidden = tf.ones(dtype=tf.float32, shape=output_shape)
    prev_cell = tf.ones(dtype=tf.float32, shape=output_shape)

    # Test we can successfully create the LSTM with partitioners.
    lstm = lstm_class(
        input_shape=input_shape[1:],
        output_channels=output_channels,
        kernel_shape=1,
        use_bias=use_bias,
        legacy_bias_behaviour=legacy_bias_behaviour,
        partitioners=partitioners)
    lstm(inputs, (prev_hidden, prev_cell))

    # Test that the variables are partitioned.
    for conv_key, convolution in lstm.convolutions.items():
      for key in keys:
        if not legacy_bias_behaviour and conv_key == "hidden" and key == "b":
          self.assertFalse(hasattr(convolution, key))
          continue
        self.assertEqual(type(getattr(convolution, key)),
                         variables.PartitionedVariable)

  @parameterized.parameters(
      (snt.Conv1DLSTM, 1, False, False),
      (snt.Conv1DLSTM, 1, True, False),
      (snt.Conv2DLSTM, 2, False, False),
      (snt.Conv2DLSTM, 2, True, False),
      (snt.Conv1DLSTM, 1, False, True),
      (snt.Conv1DLSTM, 1, True, True),
      (snt.Conv2DLSTM, 2, False, True),
      (snt.Conv2DLSTM, 2, True, True),
  )
  def testRegularizers(self, lstm_class, dim, use_bias, legacy_bias_behaviour):
    keys = snt.Conv2DLSTM.get_possible_initializer_keys(use_bias)

    batch_size = 2
    input_shape = (8,) * dim
    input_channels = 3
    output_channels = 5

    input_shape = (batch_size,) + input_shape + (input_channels,)
    output_shape = input_shape[:-1] + (output_channels,)

    inputs = tf.ones(dtype=tf.float32, shape=input_shape)
    prev_hidden = tf.ones(dtype=tf.float32, shape=output_shape)
    prev_cell = tf.ones(dtype=tf.float32, shape=output_shape)

    # Test we can successfully create the LSTM with partitioners.
    lstm = lstm_class(
        input_shape=input_shape[1:],
        output_channels=output_channels,
        kernel_shape=1,
        use_bias=use_bias,
        legacy_bias_behaviour=legacy_bias_behaviour,
        regularizers={key: tf.nn.l2_loss for key in keys})
    lstm(inputs, (prev_hidden, prev_cell))

    # Test that we have regularization losses.
    num_reg_losses = len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    num_reg_losses_expected = len(lstm.convolutions) * len(keys)
    if use_bias and not legacy_bias_behaviour:
      # Bias is not applied to hidden
      num_reg_losses_expected -= 1
    self.assertEqual(num_reg_losses, num_reg_losses_expected)

  @parameterized.parameters(
      (snt.Conv1DLSTM, 1, False),
      (snt.Conv1DLSTM, 1, True),
      (snt.Conv2DLSTM, 2, False),
      (snt.Conv2DLSTM, 2, True),
  )
  def testTraining(self, lstm_class, dim, trainable_initial_state):
    """Test that training works, with or without trainable initial state."""
    time_steps = 1
    batch_size = 2
    input_shape = (8,) * dim
    input_channels = 3
    output_channels = 5

    input_shape = (batch_size,) + input_shape + (input_channels,)

    lstm = lstm_class(
        input_shape=input_shape[1:],
        output_channels=output_channels,
        kernel_shape=1)
    inputs = tf.random_normal((time_steps,) + input_shape, dtype=tf.float32)
    initial_state = lstm.initial_state(
        batch_size, tf.float32, trainable_initial_state)

    def loss_fn():
      output, _ = tf.nn.dynamic_rnn(lstm,
                                    inputs,
                                    time_major=True,
                                    initial_state=initial_state,
                                    dtype=tf.float32)

      return tf.reduce_mean(tf.square(output))

    train_op = tf.train.GradientDescentOptimizer(1).minimize(
        loss_fn if tf.executing_eagerly() else loss_fn())
    init = tf.global_variables_initializer()
    self.evaluate(init)
    self.evaluate(train_op)

  @parameterized.parameters(
      (snt.Conv1DLSTM, 1, False, 1, 1),
      (snt.Conv1DLSTM, 1, False, 1, 5),
      (snt.Conv1DLSTM, 1, False, 6, 1),
      (snt.Conv1DLSTM, 1, False, 6, 5),
      (snt.Conv1DLSTM, 1, True, 1, 1),
      (snt.Conv1DLSTM, 1, True, 1, 5),
      (snt.Conv1DLSTM, 1, True, 6, 1),
      (snt.Conv1DLSTM, 1, True, 6, 5),
      (snt.Conv2DLSTM, 2, False, 1, 1),
      (snt.Conv2DLSTM, 2, False, 1, 5),
      (snt.Conv2DLSTM, 2, False, 6, 1),
      (snt.Conv2DLSTM, 2, False, 6, 5),
      (snt.Conv2DLSTM, 2, True, 1, 1),
      (snt.Conv2DLSTM, 2, True, 1, 5),
      (snt.Conv2DLSTM, 2, True, 6, 1),
      (snt.Conv2DLSTM, 2, True, 6, 5),
  )
  def testDilatedConv(self, lstm_class, dim, trainable_initial_state, rate,
                      kernel_shape):
    """Test that training works, with or without dilated convolutions."""
    time_steps = 1
    batch_size = 2
    input_shape = (8,) * dim
    input_channels = 3
    output_channels = 5

    input_shape = (batch_size,) + input_shape + (input_channels,)

    lstm = lstm_class(
        input_shape=input_shape[1:],
        output_channels=output_channels,
        kernel_shape=kernel_shape,
        rate=rate)
    inputs = tf.random_normal((time_steps,) + input_shape, dtype=tf.float32)
    initial_state = lstm.initial_state(
        batch_size, tf.float32, trainable_initial_state)

    def loss_fn():
      output, _ = tf.nn.dynamic_rnn(lstm,
                                    inputs,
                                    time_major=True,
                                    initial_state=initial_state,
                                    dtype=tf.float32)

      return tf.reduce_mean(tf.square(output))

    train_op = tf.train.GradientDescentOptimizer(1).minimize(
        loss_fn if tf.executing_eagerly() else loss_fn())
    init = tf.global_variables_initializer()
    self.evaluate(init)
    self.evaluate(train_op)

  @parameterized.parameters(
      (snt.Conv1DLSTM, 1, False, False, 2),
      (snt.Conv1DLSTM, 1, False, True, 4),
      (snt.Conv1DLSTM, 1, True, False, 4),
      (snt.Conv1DLSTM, 1, True, True, 6),
      (snt.Conv2DLSTM, 2, False, False, 2),
      (snt.Conv2DLSTM, 2, False, True, 4),
      (snt.Conv2DLSTM, 2, True, False, 4),
      (snt.Conv2DLSTM, 2, True, True, 6),
  )
  def testLayerNormVariables(self, lstm_class, dim, use_bias, use_layer_norm,
                             expected_num_variables):
    batch_size = 2
    input_shape = (8,) * dim + (3,)

    lstm = lstm_class(
        input_shape=input_shape,
        output_channels=5,
        kernel_shape=3,
        use_bias=use_bias,
        use_layer_norm=use_layer_norm)
    inputs = tf.ones(dtype=tf.float32, shape=((1, batch_size) + input_shape))
    initial_state = lstm.initial_state(batch_size, tf.float32)
    tf.nn.dynamic_rnn(lstm,
                      inputs,
                      time_major=True,
                      initial_state=initial_state,
                      dtype=tf.float32)

    self.assertEqual(use_layer_norm, lstm.use_layer_norm)

    # Expect the following variables:
    # Weight, and bias if present, to apply to input
    # Weight, and bias if present, to apply to hidden state
    # LayerNorm's gamma and beta, if present
    self.assertEqual(len(lstm.get_variables()), expected_num_variables)

  @parameterized.parameters(
      (snt.Conv1DLSTM, 1, False),
      (snt.Conv1DLSTM, 1, True),
      (snt.Conv2DLSTM, 2, False),
      (snt.Conv2DLSTM, 2, True),
  )
  def testLayerNorm(self, lstm_class, dim, use_bias):
    """Test that training works, with or without dilated convolutions."""
    time_steps = 3
    batch_size = 2
    input_shape = (8,) * dim
    input_channels = 3
    output_channels = 5
    kernel_shape = 3

    input_shape = (batch_size,) + input_shape + (input_channels,)

    lstm = lstm_class(
        input_shape=input_shape[1:],
        output_channels=output_channels,
        kernel_shape=kernel_shape,
        use_bias=use_bias,
        use_layer_norm=True)
    inputs = tf.random_normal((time_steps,) + input_shape, dtype=tf.float32)
    initial_state = lstm.initial_state(batch_size, tf.float32)

    def loss_fn():
      output, _ = tf.nn.dynamic_rnn(lstm,
                                    inputs,
                                    time_major=True,
                                    initial_state=initial_state,
                                    dtype=tf.float32)

      return tf.reduce_mean(tf.square(output))

    train_op = tf.train.GradientDescentOptimizer(1).minimize(
        loss_fn if tf.executing_eagerly() else loss_fn())
    init = tf.global_variables_initializer()
    self.evaluate(init)
    self.evaluate(train_op)


# @tf.contrib.eager.run_all_tests_in_graph_and_eager_modes
class GRUTest(tf.test.TestCase):

  def testShape(self):
    batch_size = 2
    hidden_size = 4
    inputs = tf.ones(dtype=tf.float32, shape=[batch_size, hidden_size])
    state = tf.ones(dtype=tf.float32, shape=[batch_size, hidden_size])
    gru = snt.GRU(hidden_size, name="rnn")
    output, next_state = gru(inputs, state)
    shape = np.ndarray((batch_size, hidden_size))
    self.assertShapeEqual(shape, next_state)
    self.assertShapeEqual(shape, output)

  def testVariables(self):
    batch_size = 5
    input_size = 10
    hidden_size = 20
    mod_name = "rnn"
    inputs = tf.ones(dtype=tf.float32, shape=[batch_size, input_size])
    state = tf.ones(dtype=tf.float32, shape=[batch_size, hidden_size])
    gru = snt.GRU(hidden_size, name=mod_name)
    self.assertEqual(gru.scope_name, mod_name)
    with self.assertRaisesRegexp(snt.Error, "not instantiated yet"):
      gru.get_variables()
    gru(inputs, state)

    gru_variables = gru.get_variables()
    self.assertEqual(len(gru_variables), 9, "GRU should have 9 variables")
    param_map = {param.name.split("/")[-1].split(":")[0]: param
                 for param in gru_variables}
    for part in ["z", "r", "h"]:
      self.assertShapeEqual(
          np.ndarray(hidden_size), tf.convert_to_tensor(param_map["b" + part]))
      self.assertShapeEqual(
          np.ndarray((hidden_size, hidden_size)),
          tf.convert_to_tensor(param_map["u" + part]))
      self.assertShapeEqual(
          np.ndarray((input_size, hidden_size)),
          tf.convert_to_tensor(param_map["w" + part]))

  def testComputation(self):

    def sigmoid(x):
      return 1 / (1 + np.exp(-x))

    batch_size = 2
    input_size = 3
    hidden_size = 5

    # With random data, check the TF calculation matches the Numpy version.
    input_data = np.random.randn(batch_size, input_size)
    state_data = np.random.randn(batch_size, hidden_size)
    inputs = tf.constant(input_data)
    state_in = tf.constant(state_data)

    gru = snt.GRU(hidden_size, name="rnn")
    _, state = gru(inputs, state_in)
    gru_variables = gru.get_variables()
    param_map = {param.name.split("/")[-1].split(":")[0]: param
                 for param in gru_variables}

    self.evaluate(tf.global_variables_initializer())
    fetches = [
        state, param_map["wz"], param_map["uz"], param_map["bz"],
        param_map["wr"], param_map["ur"], param_map["br"], param_map["wh"],
        param_map["uh"], param_map["bh"]
    ]
    output = self.evaluate(fetches)

    state_ex, wz, uz, bz, wr, ur, br, wh, uh, bh = output
    z = sigmoid(np.dot(input_data, wz) + np.dot(state_data, uz) + bz)
    r = sigmoid(np.dot(input_data, wr) + np.dot(state_data, ur) + br)
    reset_state = r * state_data
    h_twiddle = np.tanh(np.dot(input_data, wh) + np.dot(reset_state, uh)+ bh)

    state_real = (1 - z) * state_data + z * h_twiddle

    self.assertAllClose(state_real, state_ex)

  def testInitializers(self):
    batch_size = 2
    hidden_size = 4

    # Test we can successfully create the GRU with initializers.
    keys = snt.GRU.POSSIBLE_KEYS
    initializers = {
        key: tf.constant_initializer(i) for i, key in enumerate(keys)
    }
    gru = snt.GRU(hidden_size, initializers=initializers)

    # Test we can build the GRU.
    inputs = tf.ones(dtype=tf.float32, shape=[batch_size, hidden_size])
    state = tf.ones(dtype=tf.float32, shape=[batch_size, hidden_size])
    gru(inputs, state)
    init = tf.global_variables_initializer()

    # Test that the initializers have been correctly applied.
    gru_variables = [getattr(gru, "_" + key) for key in keys]
    self.evaluate(init)
    gru_variables_v = self.evaluate(gru_variables)
    for i, gru_variable_v in enumerate(gru_variables_v):
      self.assertAllClose(gru_variable_v, i * np.ones(gru_variable_v.shape))

  def testPartitioners(self):
    if tf.executing_eagerly():
      self.skipTest("Partitioned variables are not supported in eager mode.")
    batch_size = 2
    hidden_size = 4

    # Test we can successfully create the GRU with partitioners.
    keys = snt.GRU.POSSIBLE_KEYS
    partitioners = {
        key: tf.variable_axis_size_partitioner(10) for key in keys
    }
    gru = snt.GRU(hidden_size, partitioners=partitioners)

    # Test we can build the GRU.
    inputs = tf.ones(dtype=tf.float32, shape=[batch_size, hidden_size])
    state = tf.ones(dtype=tf.float32, shape=[batch_size, hidden_size])
    gru(inputs, state)

    # Test that the variables are partitioned.
    for key in keys:
      self.assertEqual(type(getattr(gru, "_" + key)),
                       variables.PartitionedVariable)

  def testRegularizers(self):
    batch_size = 2
    hidden_size = 4

    # Test we can successfully create the GRU with regularizers.
    keys = snt.GRU.POSSIBLE_KEYS
    regularizers = {
        key: tf.nn.l2_loss for key in keys
    }
    gru = snt.GRU(hidden_size, regularizers=regularizers)

    # Test we can build the GRU.
    inputs = tf.ones(dtype=tf.float32, shape=[batch_size, hidden_size])
    state = tf.ones(dtype=tf.float32, shape=[batch_size, hidden_size])
    gru(inputs, state)

    # Test that we have regularization losses.
    self.assertEqual(len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)),
                     len(keys))


# @tf.contrib.eager.run_all_tests_in_graph_and_eager_modes
class HighwayCoreTest(tf.test.TestCase, parameterized.TestCase):

  def testShape(self):
    batch_size = 2
    hidden_size = 4
    inputs = tf.ones(dtype=tf.float32, shape=[batch_size, hidden_size])
    state = tf.ones(dtype=tf.float32, shape=[batch_size, hidden_size])
    core = snt.HighwayCore(hidden_size, num_layers=3)
    output, next_state = core(inputs, state)
    shape = np.ndarray((batch_size, hidden_size))
    self.assertShapeEqual(shape, next_state)
    self.assertShapeEqual(shape, output)

  def testVariables(self):
    batch_size = 5
    input_size = 10
    hidden_size = 20
    num_layers = 3
    mod_name = "rnn"
    inputs = tf.ones(dtype=tf.float32, shape=[batch_size, input_size])
    state = tf.ones(dtype=tf.float32, shape=[batch_size, hidden_size])
    core = snt.HighwayCore(hidden_size, num_layers, name=mod_name)
    self.assertEqual(core.scope_name, mod_name)
    with self.assertRaisesRegexp(snt.Error, "not instantiated yet"):
      core.get_variables()
    core(inputs, state)

    core_variables = core.get_variables()
    self.assertEqual(len(core_variables), 2 + 4 * num_layers)
    param_map = {param.name.split("/")[-1].split(":")[0]: param
                 for param in core_variables}
    self.assertShapeEqual(
        np.ndarray((input_size, hidden_size)),
        tf.convert_to_tensor(param_map["wt"]))
    self.assertShapeEqual(
        np.ndarray((input_size, hidden_size)),
        tf.convert_to_tensor(param_map["wh"]))
    for layer_index in xrange(num_layers):
      layer_str = str(layer_index)
      self.assertShapeEqual(
          np.ndarray(hidden_size),
          tf.convert_to_tensor(param_map["bt" + layer_str]))
      self.assertShapeEqual(
          np.ndarray(hidden_size),
          tf.convert_to_tensor(param_map["bh" + layer_str]))
      self.assertShapeEqual(
          np.ndarray((hidden_size, hidden_size)),
          tf.convert_to_tensor(param_map["wt" + layer_str]))
      self.assertShapeEqual(
          np.ndarray((hidden_size, hidden_size)),
          tf.convert_to_tensor(param_map["wh" + layer_str]))

  @parameterized.parameters(True, False)
  def testComputation(self, with_dropout):
    """Checks that the TF and numpy versions match on random data."""

    def sigmoid(x):
      return 1 / (1 + np.exp(-x))

    batch_size = 2
    input_size = 3
    hidden_size = 5
    num_layers = 2
    input_data = np.random.randn(batch_size, input_size)
    state_data = np.random.randn(batch_size, hidden_size)
    inputs = tf.constant(input_data)
    state_in = tf.constant(state_data)
    if with_dropout:
      core, test_core = snt.highway_core_with_recurrent_dropout(
          hidden_size, num_layers, keep_prob=1.0)
      initial_state = core.initial_state(batch_size, dtype=tf.float64)
      _, state = core(inputs, (state_in, initial_state[1]))
      core_variables = test_core.get_variables()
    else:
      core = snt.HighwayCore(hidden_size, num_layers, name="rnn")
      _, state = core(inputs, state_in)
      core_variables = core.get_variables()
    param_map = {param.name.split("/")[-1].split(":")[0]: param
                 for param in core_variables}

    param_names = ["wt", "wh"]
    param_names += ["wt0", "bt0", "wh0", "bh0", "wt1", "bt1", "wh1", "bh1"]

    self.evaluate(tf.global_variables_initializer())
    fetches = [state] + [param_map[name] for name in param_names]
    output = self.evaluate(fetches)

    state_ex, wt, wh, wt0, bt0, wh0, bh0, wt1, bt1, wh1, bh1 = output
    # Layer 1 computation.
    t = sigmoid(np.dot(input_data, wt) + np.dot(state_data, wt0) + bt0)
    h = np.tanh(np.dot(input_data, wh) + np.dot(state_data, wh0) + bh0)
    state_data = (1 - t) * state_data + t * h
    # Layer 2 computation.
    t = sigmoid(np.dot(state_data, wt1) + bt1)
    h = np.tanh(np.dot(state_data, wh1) + bh1)
    state_data = (1 - t) * state_data + t * h

    if with_dropout:
      state_ex = state_ex[0]
    self.assertAllClose(state_data, state_ex)


# @tf.contrib.eager.run_all_tests_in_graph_and_eager_modes
class LSTMBlockCellTest(tf.test.TestCase, parameterized.TestCase):

  def testShape(self):
    batch_size = 2
    hidden_size = 5
    inputs = tf.ones(dtype=tf.float32, shape=[batch_size, hidden_size])
    prev_hidden = tf.ones(dtype=tf.float32, shape=[batch_size, hidden_size])
    prev_cell = tf.ones(dtype=tf.float32, shape=[batch_size, hidden_size])
    lstm = snt.LSTMBlockCell(hidden_size)
    output, next_state = lstm(inputs, (prev_hidden, prev_cell))

    shape = np.ndarray((batch_size, hidden_size))

    self.assertShapeEqual(shape, next_state[0])
    self.assertShapeEqual(shape, next_state[1])
    self.assertShapeEqual(shape, output)

    self.assertEqual(hidden_size, lstm.output_size)
    self.assertEqual((hidden_size, hidden_size), lstm.state_size)

  def testVariables(self):
    batch_size = 5
    hidden_size = 20
    mod_name = "lstm_block"
    inputs = tf.ones(dtype=tf.float32, shape=[batch_size, hidden_size])
    prev_cell = tf.ones(dtype=tf.float32, shape=[batch_size, hidden_size])
    prev_hidden = tf.ones(dtype=tf.float32, shape=[batch_size, hidden_size])
    lstm = snt.LSTMBlockCell(hidden_size, name=mod_name)
    self.assertEqual(lstm.scope_name, mod_name)
    with self.assertRaisesRegexp(snt.Error, "not instantiated yet"):
      lstm.get_variables()
    lstm(inputs, (prev_hidden, prev_cell))

    lstm_variables = lstm.get_variables()
    self.assertEqual(len(lstm_variables), 2, "LSTM should have 2 variables")
    param_map = {param.name.split("/")[-1].split(":")[0]:
                 param for param in lstm_variables}

    self.assertShapeEqual(np.ndarray(4 * hidden_size),
                          tf.convert_to_tensor(param_map["bias"]))
    self.assertShapeEqual(np.ndarray((2 * hidden_size, 4 * hidden_size)),
                          tf.convert_to_tensor(param_map["kernel"]))


if __name__ == "__main__":
  tf.test.main()
