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
import numpy as np
import sonnet as snt
from sonnet.testing import parameterized
import tensorflow as tf

from tensorflow.python.ops import variables


def _get_lstm_variable_names(lstm):
  keys = lstm.get_possible_initializer_keys(
      lstm.use_peepholes,
      lstm.use_batch_norm_h,
      lstm.use_batch_norm_x,
      lstm.use_batch_norm_c)
  var_names = keys - {"w_gates", "b_gates"}
  var_names |= {"b"}
  if lstm.use_batch_norm_h or lstm.use_batch_norm_x:
    var_names |= {"w_x", "w_h"}
  else:
    var_names |= {"w_xh"}
  return var_names


def _construct_lstm(use_batch_norm_h=False, use_batch_norm_x=False,
                    use_batch_norm_c=False, **kwargs):
  # Preparing for deprecation, this uses plain LSTM if no batch norm required.
  if any([use_batch_norm_h, use_batch_norm_x, use_batch_norm_c]):
    return snt.BatchNormLSTM(
        use_batch_norm_h=use_batch_norm_h,
        use_batch_norm_x=use_batch_norm_x,
        use_batch_norm_c=use_batch_norm_c,
        **kwargs)
  else:
    return snt.LSTM(**kwargs)


def _get_possible_initializer_keys(use_peepholes, use_batch_norm_h,
                                   use_batch_norm_x, use_batch_norm_c):
  # Preparing for deprecation, this uses plain LSTM if no batch norm required.
  if any([use_batch_norm_h, use_batch_norm_x, use_batch_norm_c]):
    return snt.BatchNormLSTM.get_possible_initializer_keys(
        use_peepholes, use_batch_norm_h, use_batch_norm_x, use_batch_norm_c)
  else:
    return snt.LSTM.get_possible_initializer_keys(use_peepholes)


class LSTMTest(tf.test.TestCase, parameterized.ParameterizedTestCase):

  def testShape(self):
    batch_size = 2
    hidden_size = 4
    inputs = tf.placeholder(tf.float32, shape=[batch_size, hidden_size])
    prev_hidden = tf.placeholder(tf.float32, shape=[batch_size, hidden_size])
    prev_cell = tf.placeholder(tf.float32, shape=[batch_size, hidden_size])
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
    inputs = tf.placeholder(tf.float32, shape=[batch_size, hidden_size])
    prev_cell = tf.placeholder(tf.float32, shape=[batch_size, hidden_size])
    prev_hidden = tf.placeholder(tf.float32, shape=[batch_size, hidden_size])
    lstm = snt.LSTM(hidden_size, name=mod_name)
    self.assertEqual(lstm.scope_name, mod_name)
    with self.assertRaisesRegexp(snt.Error, "not instantiated yet"):
      lstm.get_variables()
    lstm(inputs, (prev_hidden, prev_cell))

    lstm_variables = lstm.get_variables()
    self.assertEqual(len(lstm_variables), 2, "LSTM should have 2 variables")
    param_map = {param.name.split("/")[-1].split(":")[0]:
                 param for param in lstm_variables}

    self.assertShapeEqual(np.ndarray(4 * hidden_size),
                          param_map[snt.LSTM.B_GATES].initial_value)
    self.assertShapeEqual(np.ndarray((2 * hidden_size, 4 * hidden_size)),
                          param_map[snt.LSTM.W_GATES].initial_value)

  def testComputation(self):
    batch_size = 2
    hidden_size = 4
    inputs = tf.placeholder(tf.float32, shape=[batch_size, hidden_size])
    prev_cell = tf.placeholder(tf.float32, shape=[batch_size, hidden_size])
    prev_hidden = tf.placeholder(tf.float32, shape=[batch_size, hidden_size])
    lstm = snt.LSTM(hidden_size)
    _, next_state = lstm(inputs, (prev_hidden, prev_cell))
    next_hidden, next_cell = next_state
    lstm_variables = lstm.get_variables()
    param_map = {param.name.split("/")[-1].split(":")[0]:
                 param for param in lstm_variables}

    # With random data, check the TF calculation matches the Numpy version.
    input_data = np.random.randn(batch_size, hidden_size)
    prev_hidden_data = np.random.randn(batch_size, hidden_size)
    prev_cell_data = np.random.randn(batch_size, hidden_size)

    with self.test_session() as session:
      tf.global_variables_initializer().run()
      fetches = [(next_hidden, next_cell),
                 param_map[snt.LSTM.W_GATES],
                 param_map[snt.LSTM.B_GATES]]
      output = session.run(fetches,
                           {inputs: input_data,
                            prev_cell: prev_cell_data,
                            prev_hidden: prev_hidden_data})

    next_state_ex, gate_weights_ex, gate_biases_ex = output
    in_and_hid = np.concatenate((input_data, prev_hidden_data), axis=1)
    real_gate = np.dot(in_and_hid, gate_weights_ex) + gate_biases_ex
    # i = input_gate, j = next_input, f = forget_gate, o = output_gate
    i, j, f, o = np.hsplit(real_gate, 4)
    real_cell = (prev_cell_data / (1 + np.exp(-(f + lstm._forget_bias))) +
                 1 / (1 + np.exp(-i)) * np.tanh(j))
    real_hidden = np.tanh(real_cell) * 1 / (1 + np.exp(-o))

    self.assertAllClose(real_hidden, next_state_ex[0])
    self.assertAllClose(real_cell, next_state_ex[1])

  def testPeephole(self):
    batch_size = 5
    hidden_size = 20

    # Initialize the rnn and verify the number of parameter sets.
    inputs = tf.placeholder(tf.float32, shape=[batch_size, hidden_size])
    prev_cell = tf.placeholder(tf.float32, shape=[batch_size, hidden_size])
    prev_hidden = tf.placeholder(tf.float32, shape=[batch_size, hidden_size])
    lstm = snt.LSTM(hidden_size, use_peepholes=True)
    _, next_state = lstm(inputs, (prev_hidden, prev_cell))
    next_hidden, next_cell = next_state
    lstm_variables = lstm.get_variables()
    self.assertEqual(len(lstm_variables), 5, "LSTM should have 5 variables")

    # Unpack parameters into dict and check their sizes.
    param_map = {param.name.split("/")[-1].split(":")[0]:
                 param for param in lstm_variables}
    self.assertShapeEqual(np.ndarray(4 * hidden_size),
                          param_map[snt.LSTM.B_GATES].initial_value)
    self.assertShapeEqual(np.ndarray((2 * hidden_size, 4 * hidden_size)),
                          param_map[snt.LSTM.W_GATES].initial_value)
    self.assertShapeEqual(np.ndarray(hidden_size),
                          param_map[snt.LSTM.W_F_DIAG].initial_value)
    self.assertShapeEqual(np.ndarray(hidden_size),
                          param_map[snt.LSTM.W_I_DIAG].initial_value)
    self.assertShapeEqual(np.ndarray(hidden_size),
                          param_map[snt.LSTM.W_O_DIAG].initial_value)

    # With random data, check the TF calculation matches the Numpy version.
    input_data = np.random.randn(batch_size, hidden_size)
    prev_hidden_data = np.random.randn(batch_size, hidden_size)
    prev_cell_data = np.random.randn(batch_size, hidden_size)

    with self.test_session() as session:
      tf.global_variables_initializer().run()
      fetches = [(next_hidden, next_cell),
                 param_map[snt.LSTM.W_GATES],
                 param_map[snt.LSTM.B_GATES],
                 param_map[snt.LSTM.W_F_DIAG],
                 param_map[snt.LSTM.W_I_DIAG],
                 param_map[snt.LSTM.W_O_DIAG]]
      output = session.run(fetches,
                           {inputs: input_data,
                            prev_cell: prev_cell_data,
                            prev_hidden: prev_hidden_data})

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

  @parameterized.Parameters(
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
    lstm = _construct_lstm(hidden_size=hidden_size,
                           use_peepholes=use_peepholes,
                           use_batch_norm_h=use_batch_norm_h,
                           use_batch_norm_x=use_batch_norm_x,
                           use_batch_norm_c=use_batch_norm_c,
                           initializers=initializers)

    # Test we can build the LSTM.
    inputs = tf.placeholder(tf.float32, shape=[batch_size, hidden_size])
    prev_cell = tf.placeholder(tf.float32, shape=[batch_size, hidden_size])
    prev_hidden = tf.placeholder(tf.float32, shape=[batch_size, hidden_size])
    lstm(inputs, (prev_hidden, prev_cell), is_training=True)
    init = tf.global_variables_initializer()

    # Test that the initializers have been correctly applied.
    lstm_variable_names = _get_lstm_variable_names(lstm)
    lstm_variables = [getattr(lstm, "_" + name) for name in lstm_variable_names]
    with self.test_session() as sess:
      sess.run(init)
      lstm_variables_v = sess.run(lstm_variables)
      for lstm_variable_v in lstm_variables_v:
        self.assertAllClose(lstm_variable_v,
                            1.5 * np.ones(lstm_variable_v.shape))

  def testPeepholeInitializersCheck(self):
    hidden_size = 4

    # Test that passing in a peephole initializer when we don't request peephole
    # connections raises an error.
    for key in [snt.LSTM.W_F_DIAG, snt.LSTM.W_I_DIAG, snt.LSTM.W_O_DIAG]:
      with self.assertRaisesRegexp(KeyError, "Invalid initializer"):
        snt.LSTM(hidden_size, use_peepholes=False,
                 initializers={key: tf.constant_initializer(0)})

  @parameterized.Parameters(
      (True, False, False),
      (False, True, False),
      (False, False, True)
  )
  def testBatchNormBuildFlag(self, use_batch_norm_h, use_batch_norm_x,
                             use_batch_norm_c):
    """Check if an error is raised if we don't specify the is_training flag."""
    batch_size = 2
    hidden_size = 4

    inputs = tf.placeholder(tf.float32, shape=[batch_size, hidden_size])
    prev_cell = tf.placeholder(tf.float32, shape=[batch_size, hidden_size])
    prev_hidden = tf.placeholder(tf.float32, shape=[batch_size, hidden_size])

    err = "is_training flag must be explicitly specified"
    with self.assertRaisesRegexp(ValueError, err):
      lstm = snt.LSTM(hidden_size,
                      use_batch_norm_h=use_batch_norm_h,
                      use_batch_norm_x=use_batch_norm_x,
                      use_batch_norm_c=use_batch_norm_c)
      lstm(inputs, (prev_cell, prev_hidden))

  def testBatchNormInitializersCheck(self):
    hidden_size = 4

    # N.B. batch norm options to LSTM are deprecated now, so this is just
    # checking that bad uses of deprecated options are complained about.

    # Test that passing in a batchnorm initializer when we don't request
    # batchnorm raises an error.
    for key in [snt.LSTM.GAMMA_H, snt.LSTM.GAMMA_X, snt.LSTM.GAMMA_C,
                snt.LSTM.BETA_C]:
      with self.assertRaisesRegexp(KeyError, "Invalid initializer"):
        snt.LSTM(hidden_size,
                 initializers={key: tf.constant_initializer(0)})

    # Test that setting max_unique_stats=2 without batchnorm raises an error.
    with self.assertRaisesRegexp(ValueError, "max_unique_stats specified.*"):
      snt.LSTM(hidden_size, max_unique_stats=2)

  @parameterized.Parameters(
      *itertools.product(
          (True, False), (True, False), (True, False), (True, False))
  )
  def testPartitioners(self, use_peepholes, use_batch_norm_h, use_batch_norm_x,
                       use_batch_norm_c):
    batch_size = 2
    hidden_size = 4

    keys = _get_possible_initializer_keys(
        use_peepholes, use_batch_norm_h, use_batch_norm_x, use_batch_norm_c)
    partitioners = {
        key: tf.variable_axis_size_partitioner(10) for key in keys
    }

    # Test we can successfully create the LSTM with partitioners.
    lstm = _construct_lstm(hidden_size=hidden_size,
                           use_peepholes=use_peepholes,
                           use_batch_norm_h=use_batch_norm_h,
                           use_batch_norm_x=use_batch_norm_x,
                           use_batch_norm_c=use_batch_norm_c,
                           partitioners=partitioners)

    # Test we can build the LSTM
    inputs = tf.placeholder(tf.float32, shape=[batch_size, hidden_size])
    prev_cell = tf.placeholder(tf.float32, shape=[batch_size, hidden_size])
    prev_hidden = tf.placeholder(tf.float32, shape=[batch_size, hidden_size])
    lstm(inputs, (prev_hidden, prev_cell), is_training=True)

    # Test that the variables are partitioned.
    var_names = _get_lstm_variable_names(lstm)
    for var_name in var_names:
      self.assertEqual(type(getattr(lstm, "_" + var_name)),
                       variables.PartitionedVariable)

  @parameterized.Parameters(
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
    lstm = _construct_lstm(hidden_size=hidden_size,
                           use_peepholes=use_peepholes,
                           use_batch_norm_h=use_batch_norm_h,
                           use_batch_norm_x=use_batch_norm_x,
                           use_batch_norm_c=use_batch_norm_c,
                           regularizers=regularizers)

    # Test we can build the LSTM
    inputs = tf.placeholder(tf.float32, shape=[batch_size, hidden_size])
    prev_cell = tf.placeholder(tf.float32, shape=[batch_size, hidden_size])
    prev_hidden = tf.placeholder(tf.float32, shape=[batch_size, hidden_size])
    lstm(inputs, (prev_hidden, prev_cell), is_training=True)

    # Test that we have regularization losses.
    num_reg_losses = len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    if use_batch_norm_h or use_batch_norm_x:
      self.assertEqual(num_reg_losses, len(keys) + 1)
    else:
      self.assertEqual(num_reg_losses, len(keys))

  # Pick some hopefully representative combination of parameter values
  # (want to test with and without BatchNorm, and with
  # seq_len < max_unique_stats and seq_len > max_unique_stats, and some other
  # combinations for good measure).
  @parameterized.Parameters(
      (False, False, 3, 1, 2),
      (False, True, 1, 1, 2),
      (True, True, 3, 1, 2),
      (False, True, 1, 2, 1),
      (True, True, 3, 2, 1),
      (False, True, 3, 3, 5))
  def testSameInStaticAndDynamic(self, use_peepholes, use_batch_norm,
                                 batch_size, max_unique_stats, seq_len):
    # Tests that when the cell is used in either a normal tensorflow rnn, or in
    # tensorflow's dynamic_rnn, that the output is the same. This is to test
    # test that the cores aren't doing anything funny they shouldn't be (like
    # relying on the number of times they've been invoked).

    hidden_size = 3
    input_size = 3

    inputs = tf.placeholder(tf.float32,
                            shape=[batch_size, seq_len, input_size],
                            name="inputs")
    static_inputs = tf.unstack(inputs, axis=1)

    test_local_stats = False

    cell = _construct_lstm(hidden_size=hidden_size,
                           max_unique_stats=max_unique_stats,
                           use_peepholes=use_peepholes,
                           use_batch_norm_h=use_batch_norm,
                           use_batch_norm_x=use_batch_norm,
                           use_batch_norm_c=use_batch_norm)

    # Connect static in training and test modes
    train_static_output_unpacked, _ = tf.contrib.rnn.static_rnn(
        cell.with_batch_norm_control(is_training=True,
                                     test_local_stats=test_local_stats),
        static_inputs,
        initial_state=cell.initial_state(batch_size, tf.float32))

    test_static_output_unpacked, _ = tf.contrib.rnn.static_rnn(
        cell.with_batch_norm_control(is_training=False,
                                     test_local_stats=test_local_stats),
        static_inputs,
        initial_state=cell.initial_state(batch_size, tf.float32))

    # Connect dynamic in training and test modes
    train_dynamic_output, _ = tf.nn.dynamic_rnn(
        cell.with_batch_norm_control(is_training=True,
                                     test_local_stats=test_local_stats),
        inputs,
        initial_state=cell.initial_state(batch_size, tf.float32),
        dtype=tf.float32)

    test_dynamic_output, _ = tf.nn.dynamic_rnn(
        cell.with_batch_norm_control(is_training=False,
                                     test_local_stats=test_local_stats),
        inputs,
        initial_state=cell.initial_state(batch_size, tf.float32),
        dtype=tf.float32)

    train_static_output = tf.stack(train_static_output_unpacked, axis=1)
    test_static_output = tf.stack(test_static_output_unpacked, axis=1)

    with self.test_session() as session:
      tf.global_variables_initializer().run()

      def check_static_and_dynamic(training):
        # Check that static and dynamic give the same output
        input_data = np.random.rand(batch_size, seq_len, input_size)

        if training:
          ops = [train_static_output, train_dynamic_output]
        else:
          ops = [test_static_output, test_dynamic_output]

        static_out, dynamic_out = session.run(ops,
                                              feed_dict={inputs: input_data})
        self.assertAllClose(static_out, dynamic_out)

      # Do a pass to train the exponential moving statistics.
      for _ in range(5):
        check_static_and_dynamic(True)

      # And check that same when using test statistics.
      check_static_and_dynamic(False)

  def testLayerNormVariables(self):
    core = snt.LSTM(hidden_size=3, use_layer_norm=True)

    batch_size = 3
    inputs = tf.placeholder(tf.float32, shape=[batch_size, 3, 3])
    tf.nn.dynamic_rnn(core,
                      inputs,
                      initial_state=core.initial_state(batch_size, tf.float32))

    self.assertTrue(core.use_layer_norm)

    expected = 4  # gate bias and one weight, plus LayerNorm's gamma, beta.
    self.assertEqual(len(core.get_variables()), expected)

  def testHiddenClipping(self):
    core = snt.LSTM(hidden_size=5, hidden_clip_value=1.0)
    obs = tf.constant(np.random.rand(3, 10), dtype=tf.float32)
    hidden = tf.placeholder(tf.float32, shape=[3, 5])
    cell = tf.placeholder(tf.float32, shape=[3, 5])
    output = core(obs, [hidden, cell])
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      unclipped = np.random.rand(3, 5) - 0.5
      unclipped *= 2.0 / unclipped.max()
      clipped = unclipped.clip(-1., 1.)
      output1, (hidden1, cell1) = sess.run(output, feed_dict={hidden: unclipped,
                                                              cell: unclipped})
      output2, (hidden2, cell2) = sess.run(output, feed_dict={hidden: clipped,
                                                              cell: unclipped})
      self.assertAllClose(output1, output2)
      self.assertAllClose(hidden1, hidden2)
      self.assertAllClose(cell1, cell2)

  def testCellClipping(self):
    core = snt.LSTM(hidden_size=5, cell_clip_value=1.0)
    obs = tf.constant(np.random.rand(3, 10), dtype=tf.float32)
    hidden = tf.placeholder(tf.float32, shape=[3, 5])
    cell = tf.placeholder(tf.float32, shape=[3, 5])
    output = core(obs, [hidden, cell])
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      unclipped = np.random.rand(3, 5) - 0.5
      unclipped *= 2.0 / unclipped.max()
      clipped = unclipped.clip(-1., 1.)

      output1, (hidden1, cell1) = sess.run(output, feed_dict={hidden: unclipped,
                                                              cell: unclipped})
      output2, (hidden2, cell2) = sess.run(output, feed_dict={hidden: unclipped,
                                                              cell: clipped})
      self.assertAllClose(output1, output2)
      self.assertAllClose(hidden1, hidden2)
      self.assertAllClose(cell1, cell2)

  def testConflictingNormalization(self):
    # N.B. batch norm options to LSTM are deprecated now, so this is just
    # checking that bad uses of deprecated options are complained about.

    with self.assertRaisesRegexp(
        ValueError, "Only one of use_batch_norm_h and layer_norm is allowed."):
      snt.LSTM(hidden_size=3, use_layer_norm=True, use_batch_norm_h=True)

    with self.assertRaisesRegexp(
        ValueError, "Only one of use_batch_norm_x and layer_norm is allowed."):
      snt.LSTM(hidden_size=3, use_layer_norm=True, use_batch_norm_x=True)

    with self.assertRaisesRegexp(
        ValueError, "Only one of use_batch_norm_c and layer_norm is allowed."):
      snt.LSTM(hidden_size=3, use_layer_norm=True, use_batch_norm_c=True)

  @parameterized.Parameters(
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
    cell = _construct_lstm(hidden_size=3,
                           use_peepholes=use_peepholes,
                           use_batch_norm_h=use_batch_norm_h,
                           use_batch_norm_x=use_batch_norm_x,
                           use_batch_norm_c=use_batch_norm_c)

    # Need to connect the cell before it has variables
    batch_size = 3
    inputs = tf.placeholder(tf.float32, shape=[batch_size, 3, 3])
    tf.nn.dynamic_rnn(cell.with_batch_norm_control(is_training=True),
                      inputs,
                      initial_state=cell.initial_state(batch_size, tf.float32))

    self.assertEqual(use_peepholes, cell.use_peepholes)
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

  @parameterized.Parameters(
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
    output, _ = tf.nn.dynamic_rnn(
        cell.with_batch_norm_control(is_training=True),
        inputs,
        initial_state=initial_state,
        dtype=tf.float32)

    loss = tf.reduce_mean(tf.square(
        output - np.random.rand(batch_size, time_steps, hidden_size)))
    train_op = tf.train.GradientDescentOptimizer(1).minimize(loss)
    init = tf.global_variables_initializer()
    with self.test_session():
      init.run()
      train_op.run()

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
    lstm = snt.LSTM(hidden_size=3, name="foo")
    unnamed_init_state = lstm.initial_state(4, trainable=True)
    named_init_state = lstm.initial_state(4, trainable=True, name="bar")
    self.assertEqual(unnamed_init_state[0].name,
                     "foo_initial_state/state_0_tiled:0")
    self.assertEqual(unnamed_init_state[1].name,
                     "foo_initial_state/state_1_tiled:0")
    self.assertEqual(named_init_state[0].name, "bar/state_0_tiled:0")
    self.assertEqual(named_init_state[1].name, "bar/state_1_tiled:0")


class ConvLSTMTest(tf.test.TestCase, parameterized.ParameterizedTestCase):

  @parameterized.Parameters(
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

    inputs = tf.placeholder(tf.float32, shape=input_shape)
    prev_hidden = tf.placeholder(tf.float32, shape=output_shape)
    prev_cell = tf.placeholder(tf.float32, shape=output_shape)
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

  @parameterized.Parameters(
      (snt.Conv1DLSTM, 1, False),
      (snt.Conv1DLSTM, 1, True),
      (snt.Conv2DLSTM, 2, False),
      (snt.Conv2DLSTM, 2, True),
  )
  def testInitializers(self, lstm_class, dim, use_bias):
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

    inputs = tf.placeholder(tf.float32, shape=input_shape)
    prev_hidden = tf.placeholder(tf.float32, shape=output_shape)
    prev_cell = tf.placeholder(tf.float32, shape=output_shape)

    # Test we can successfully create the LSTM with partitioners.
    lstm = lstm_class(
        input_shape=input_shape[1:],
        output_channels=output_channels,
        kernel_shape=1,
        use_bias=use_bias,
        initializers=initializers)
    lstm(inputs, (prev_hidden, prev_cell))

    init = tf.global_variables_initializer()

    # Test that the initializers have been applied correctly.
    with self.test_session() as sess:
      sess.run(init)
      for convolution in lstm.convolutions.values():
        for i, key in enumerate(keys):
          variable = getattr(convolution, key)
          self.assertAllClose(sess.run(variable),
                              np.full(variable.get_shape(), i))

  @parameterized.Parameters(
      (snt.Conv1DLSTM, 1, False),
      (snt.Conv1DLSTM, 1, True),
      (snt.Conv2DLSTM, 2, False),
      (snt.Conv2DLSTM, 2, True),
  )
  def testPartitioners(self, lstm_class, dim, use_bias):
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

    inputs = tf.placeholder(tf.float32, shape=input_shape)
    prev_hidden = tf.placeholder(tf.float32, shape=output_shape)
    prev_cell = tf.placeholder(tf.float32, shape=output_shape)

    # Test we can successfully create the LSTM with partitioners.
    lstm = lstm_class(
        input_shape=input_shape[1:],
        output_channels=output_channels,
        kernel_shape=1,
        use_bias=use_bias,
        partitioners=partitioners)
    lstm(inputs, (prev_hidden, prev_cell))

    # Test that the variables are partitioned.
    for convolution in lstm.convolutions.values():
      for key in keys:
        self.assertEqual(type(getattr(convolution, key)),
                         variables.PartitionedVariable)

  @parameterized.Parameters(
      (snt.Conv1DLSTM, 1, False),
      (snt.Conv1DLSTM, 1, True),
      (snt.Conv2DLSTM, 2, False),
      (snt.Conv2DLSTM, 2, True),
  )
  def testRegularizers(self, lstm_class, dim, use_bias):
    keys = snt.Conv2DLSTM.get_possible_initializer_keys(use_bias)

    batch_size = 2
    input_shape = (8,) * dim
    input_channels = 3
    output_channels = 5

    input_shape = (batch_size,) + input_shape + (input_channels,)
    output_shape = input_shape[:-1] + (output_channels,)

    inputs = tf.placeholder(tf.float32, shape=input_shape)
    prev_hidden = tf.placeholder(tf.float32, shape=output_shape)
    prev_cell = tf.placeholder(tf.float32, shape=output_shape)

    # Test we can successfully create the LSTM with partitioners.
    lstm = lstm_class(
        input_shape=input_shape[1:],
        output_channels=output_channels,
        kernel_shape=1,
        use_bias=use_bias,
        regularizers={key: tf.nn.l2_loss for key in keys})
    lstm(inputs, (prev_hidden, prev_cell))

    # Test that we have regularization losses.
    num_reg_losses = len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    self.assertEqual(num_reg_losses, len(lstm.convolutions) * len(keys))

  @parameterized.Parameters(
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

    output, _ = tf.nn.dynamic_rnn(lstm,
                                  inputs,
                                  time_major=True,
                                  initial_state=initial_state,
                                  dtype=tf.float32)

    loss = tf.reduce_mean(tf.square(output))
    train_op = tf.train.GradientDescentOptimizer(1).minimize(loss)
    init = tf.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(init)
      sess.run(train_op)


class GRUTest(tf.test.TestCase):

  def testShape(self):
    batch_size = 2
    hidden_size = 4
    inputs = tf.placeholder(tf.float32, shape=[batch_size, hidden_size])
    state = tf.placeholder(tf.float32, shape=[batch_size, hidden_size])
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
    inputs = tf.placeholder(tf.float32, shape=[batch_size, input_size])
    state = tf.placeholder(tf.float32, shape=[batch_size, hidden_size])
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
      self.assertShapeEqual(np.ndarray(hidden_size),
                            param_map["b" + part].initial_value)
      self.assertShapeEqual(np.ndarray((hidden_size, hidden_size)),
                            param_map["u" + part].initial_value)
      self.assertShapeEqual(np.ndarray((input_size, hidden_size)),
                            param_map["w" + part].initial_value)

  def testComputation(self):

    def sigmoid(x):
      return 1 / (1 + np.exp(-x))

    batch_size = 2
    input_size = 3
    hidden_size = 5
    inputs = tf.placeholder(tf.float64, shape=[batch_size, input_size])
    state_in = tf.placeholder(tf.float64, shape=[batch_size, hidden_size])
    gru = snt.GRU(hidden_size, name="rnn")
    _, state = gru(inputs, state_in)
    gru_variables = gru.get_variables()
    param_map = {param.name.split("/")[-1].split(":")[0]: param
                 for param in gru_variables}

    # With random data, check the TF calculation matches the Numpy version.
    input_data = np.random.randn(batch_size, input_size)
    state_data = np.random.randn(batch_size, hidden_size)

    with self.test_session() as session:
      tf.global_variables_initializer().run()
      fetches = [state, param_map["wz"], param_map["uz"], param_map["bz"],
                 param_map["wr"], param_map["ur"], param_map["br"],
                 param_map["wh"], param_map["uh"], param_map["bh"]]
      output = session.run(fetches, {inputs: input_data, state_in: state_data})

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
    inputs = tf.placeholder(tf.float32, shape=[batch_size, hidden_size])
    state = tf.placeholder(tf.float32, shape=[batch_size, hidden_size])
    gru(inputs, state)
    init = tf.global_variables_initializer()

    # Test that the initializers have been correctly applied.
    gru_variables = [getattr(gru, "_" + key) for key in keys]
    with self.test_session() as sess:
      sess.run(init)
      gru_variables_v = sess.run(gru_variables)
      for i, gru_variable_v in enumerate(gru_variables_v):
        self.assertAllClose(gru_variable_v,
                            i * np.ones(gru_variable_v.shape))

  def testPartitioners(self):
    batch_size = 2
    hidden_size = 4

    # Test we can successfully create the GRU with partitioners.
    keys = snt.GRU.POSSIBLE_KEYS
    partitioners = {
        key: tf.variable_axis_size_partitioner(10) for key in keys
    }
    gru = snt.GRU(hidden_size, partitioners=partitioners)

    # Test we can build the GRU.
    inputs = tf.placeholder(tf.float32, shape=[batch_size, hidden_size])
    state = tf.placeholder(tf.float32, shape=[batch_size, hidden_size])
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
    inputs = tf.placeholder(tf.float32, shape=[batch_size, hidden_size])
    state = tf.placeholder(tf.float32, shape=[batch_size, hidden_size])
    gru(inputs, state)

    # Test that we have regularization losses.
    self.assertEqual(len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)),
                     len(keys))


if __name__ == "__main__":
  tf.test.main()
