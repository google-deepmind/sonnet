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

"""Tests for Recurrent cores in snt."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

# Dependency imports
import mock
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import sonnet as snt
from sonnet.testing import parameterized
import tensorflow as tf

from tensorflow.python.ops import variables


class VanillaRNNTest(tf.test.TestCase):

  def setUp(self):
    super(VanillaRNNTest, self).setUp()
    self.batch_size = 3
    self.in_size = 4
    self.hidden_size = 18

  def testShape(self):
    inputs = tf.placeholder(tf.float32, shape=[self.batch_size, self.in_size])
    prev_state = tf.placeholder(tf.float32,
                                shape=[self.batch_size, self.hidden_size])
    vanilla_rnn = snt.VanillaRNN(name="rnn", hidden_size=self.hidden_size)
    output, next_state = vanilla_rnn(inputs, prev_state)
    shape = np.ndarray((self.batch_size, self.hidden_size))
    self.assertShapeEqual(shape, output)
    self.assertShapeEqual(shape, next_state)

  def testVariables(self):
    mod_name = "rnn"
    inputs = tf.placeholder(tf.float32, shape=[self.batch_size, self.in_size])
    prev_state = tf.placeholder(tf.float32,
                                shape=[self.batch_size, self.hidden_size])
    vanilla_rnn = snt.VanillaRNN(name=mod_name, hidden_size=self.hidden_size)
    self.assertEqual(vanilla_rnn.scope_name, mod_name)
    with self.assertRaisesRegexp(snt.Error, "not instantiated yet"):
      vanilla_rnn.get_variables()
    vanilla_rnn(inputs, prev_state)

    rnn_variables = vanilla_rnn.get_variables()
    self.assertEqual(len(rnn_variables), 4, "RNN should have 4 variables")

    in_to_hidden_w = next(
        v for v in rnn_variables if v.name == "%s/in_to_hidden/w:0" % mod_name)
    in_to_hidden_b = next(
        v for v in rnn_variables if v.name == "%s/in_to_hidden/b:0" % mod_name)
    hidden_to_hidden_w = next(
        v for v in rnn_variables
        if v.name == "%s/hidden_to_hidden/w:0" % mod_name)
    hidden_to_hidden_b = next(
        v for v in rnn_variables
        if v.name == "%s/hidden_to_hidden/b:0" % mod_name)
    self.assertShapeEqual(np.ndarray((self.in_size, self.hidden_size)),
                          in_to_hidden_w.initial_value)
    self.assertShapeEqual(np.ndarray(self.hidden_size),
                          in_to_hidden_b.initial_value)
    self.assertShapeEqual(np.ndarray((self.hidden_size, self.hidden_size)),
                          hidden_to_hidden_w.initial_value)
    self.assertShapeEqual(np.ndarray(self.hidden_size),
                          hidden_to_hidden_b.initial_value)

  def testComputation(self):
    inputs = tf.placeholder(tf.float32, shape=[self.batch_size, self.in_size])
    prev_state = tf.placeholder(tf.float32,
                                shape=[self.batch_size, self.hidden_size])
    vanilla_rnn = snt.VanillaRNN(name="rnn", hidden_size=self.hidden_size)
    output, next_state = vanilla_rnn(inputs, prev_state)
    in_to_hid = vanilla_rnn.in_to_hidden_variables
    hid_to_hid = vanilla_rnn.hidden_to_hidden_variables
    with self.test_session() as sess:
      # With random data, check the TF calculation matches the Numpy version.
      input_data = np.random.randn(self.batch_size, self.in_size)
      prev_state_data = np.random.randn(self.batch_size, self.hidden_size)
      tf.global_variables_initializer().run()

      fetches = [output, next_state, in_to_hid[0], in_to_hid[1],
                 hid_to_hid[0], hid_to_hid[1]]
      output = sess.run(fetches,
                        {inputs: input_data, prev_state: prev_state_data})
    output_v, next_state_v, in_to_hid_w, in_to_hid_b = output[:4]
    hid_to_hid_w, hid_to_hid_b = output[4:]

    real_in_to_hid = np.dot(input_data, in_to_hid_w) + in_to_hid_b
    real_hid_to_hid = np.dot(prev_state_data, hid_to_hid_w) + hid_to_hid_b
    real_output = np.tanh(real_in_to_hid + real_hid_to_hid)

    self.assertAllClose(real_output, output_v)
    self.assertAllClose(real_output, next_state_v)

  def testInitializers(self):
    inputs = tf.placeholder(tf.float32, shape=[self.batch_size, self.in_size])
    prev_state = tf.placeholder(tf.float32,
                                shape=[self.batch_size, self.hidden_size])

    with self.assertRaisesRegexp(KeyError, "Invalid initializer keys.*"):
      snt.VanillaRNN(name="rnn",
                     hidden_size=self.hidden_size,
                     initializers={"invalid": None})

    err = "Initializer for 'w' is not a callable function"
    with self.assertRaisesRegexp(TypeError, err):
      snt.VanillaRNN(name="rnn",
                     hidden_size=self.hidden_size,
                     initializers={"in_to_hidden": {"w": tf.zeros([10, 10])}})

    # Nested initializer.
    valid_initializers = {
        "in_to_hidden": {
            "w": tf.ones_initializer(),
        },
        "hidden_to_hidden": {
            "b": tf.ones_initializer(),
        }
    }

    vanilla_rnn = snt.VanillaRNN(name="rnn",
                                 hidden_size=self.hidden_size,
                                 initializers=valid_initializers)

    vanilla_rnn(inputs, prev_state)
    init = tf.global_variables_initializer()

    with self.test_session() as sess:
      sess.run(init)
      w_v, b_v = sess.run([
          vanilla_rnn.in_to_hidden_linear.w,
          vanilla_rnn.hidden_to_hidden_linear.b,
      ])
      self.assertAllClose(w_v, np.ones([self.in_size, self.hidden_size]))
      self.assertAllClose(b_v, np.ones([self.hidden_size]))

  def testPartitioners(self):
    inputs = tf.placeholder(tf.float32, shape=[self.batch_size, self.in_size])
    prev_state = tf.placeholder(tf.float32,
                                shape=[self.batch_size, self.hidden_size])

    with self.assertRaisesRegexp(KeyError, "Invalid partitioner keys.*"):
      snt.VanillaRNN(name="rnn",
                     hidden_size=self.hidden_size,
                     partitioners={"invalid": None})

    err = "Partitioner for 'w' is not a callable function"
    with self.assertRaisesRegexp(TypeError, err):
      snt.VanillaRNN(name="rnn",
                     hidden_size=self.hidden_size,
                     partitioners={"in_to_hidden": {"w": tf.zeros([10, 10])}})

    # Nested partitioners.
    valid_partitioners = {
        "in_to_hidden": {
            "w": tf.fixed_size_partitioner(num_shards=2),
            "b": tf.fixed_size_partitioner(num_shards=2),
        },
        "hidden_to_hidden": {
            "w": tf.fixed_size_partitioner(num_shards=2),
            "b": tf.fixed_size_partitioner(num_shards=2),
        }
    }

    vanilla_rnn = snt.VanillaRNN(name="rnn",
                                 hidden_size=self.hidden_size,
                                 partitioners=valid_partitioners)

    vanilla_rnn(inputs, prev_state)

    self.assertEqual(type(vanilla_rnn.in_to_hidden_linear.w),
                     variables.PartitionedVariable)
    self.assertEqual(type(vanilla_rnn.in_to_hidden_linear.b),
                     variables.PartitionedVariable)
    self.assertEqual(type(vanilla_rnn.hidden_to_hidden_linear.w),
                     variables.PartitionedVariable)
    self.assertEqual(type(vanilla_rnn.hidden_to_hidden_linear.b),
                     variables.PartitionedVariable)

  def testRegularizers(self):
    inputs = tf.placeholder(tf.float32, shape=[self.batch_size, self.in_size])
    prev_state = tf.placeholder(tf.float32,
                                shape=[self.batch_size, self.hidden_size])

    with self.assertRaisesRegexp(KeyError, "Invalid regularizer keys.*"):
      snt.VanillaRNN(name="rnn",
                     hidden_size=self.hidden_size,
                     regularizers={"invalid": None})

    err = "Regularizer for 'w' is not a callable function"
    with self.assertRaisesRegexp(TypeError, err):
      snt.VanillaRNN(name="rnn",
                     hidden_size=self.hidden_size,
                     regularizers={"in_to_hidden": {"w": tf.zeros([10, 10])}})

    # Nested regularizers.
    valid_regularizers = {
        "in_to_hidden": {
            "w": tf.nn.l2_loss,
        },
        "hidden_to_hidden": {
            "b": tf.nn.l2_loss,
        }
    }

    vanilla_rnn = snt.VanillaRNN(name="rnn",
                                 hidden_size=self.hidden_size,
                                 regularizers=valid_regularizers)
    vanilla_rnn(inputs, prev_state)
    regularizers = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    self.assertEqual(len(regularizers), 2)


class DeepRNNTest(tf.test.TestCase, parameterized.ParameterizedTestCase):

  def testShape(self):
    batch_size = 3
    batch_size_shape = tf.TensorShape(batch_size)
    in_size = 2
    hidden1_size = 4
    hidden2_size = 5

    inputs = tf.placeholder(tf.float32, shape=[batch_size, in_size])
    prev_state0 = tf.placeholder(tf.float32, shape=[batch_size, in_size])
    prev_state1 = tf.placeholder(tf.float32, shape=[batch_size, hidden1_size])
    prev_state2 = tf.placeholder(tf.float32, shape=[batch_size, hidden2_size])
    prev_state = (prev_state0, prev_state1, prev_state2)

    # Test recurrent and non-recurrent cores
    cores = [snt.VanillaRNN(name="rnn0", hidden_size=in_size),
             snt.VanillaRNN(name="rnn1", hidden_size=hidden1_size),
             snt.VanillaRNN(name="rnn2", hidden_size=hidden2_size)]
    deep_rnn = snt.DeepRNN(cores, name="deep_rnn",
                           skip_connections=True)
    output, next_state = deep_rnn(inputs, prev_state)
    output_shape = output.get_shape()

    output_size = in_size + hidden1_size + hidden2_size

    self.assertTrue(
        output_shape.is_compatible_with([batch_size, output_size]))
    self.assertTrue(output_shape.is_compatible_with(
        batch_size_shape.concatenate(deep_rnn.output_size)))

    next_state_shape = (next_state[0].get_shape(), next_state[1].get_shape(),
                        next_state[2].get_shape())

    self.assertTrue(
        next_state_shape[0].is_compatible_with([batch_size, in_size]))
    self.assertTrue(
        next_state_shape[1].is_compatible_with([batch_size, hidden1_size]))
    self.assertTrue(
        next_state_shape[2].is_compatible_with([batch_size, hidden2_size]))

    for state_shape, expected_shape in zip(next_state_shape,
                                           deep_rnn.state_size):
      self.assertTrue(state_shape.is_compatible_with(
          batch_size_shape.concatenate(expected_shape)))

    # Initial state should be a valid state
    initial_state = deep_rnn.initial_state(batch_size, tf.float32)
    self.assertTrue(len(initial_state), len(next_state))
    self.assertShapeEqual(np.ndarray((batch_size, in_size)),
                          initial_state[0])
    self.assertShapeEqual(np.ndarray((batch_size, hidden1_size)),
                          initial_state[1])
    self.assertShapeEqual(np.ndarray((batch_size, hidden2_size)),
                          initial_state[2])

  def testIncompatibleOptions(self):
    in_size = 2
    hidden1_size = 4
    hidden2_size = 5
    cores = [snt.Linear(name="linear", output_size=in_size),
             snt.VanillaRNN(name="rnn1", hidden_size=hidden1_size),
             snt.VanillaRNN(name="rnn2", hidden_size=hidden2_size)]
    with self.assertRaisesRegexp(
        ValueError, "skip_connections are enabled but not all cores are "
                    "`snt.RNNCore`s, which is not supported"):
      snt.DeepRNN(cores, name="deep_rnn", skip_connections=True)

    cells = [tf.contrib.rnn.BasicLSTMCell(5), tf.contrib.rnn.BasicLSTMCell(5)]
    with self.assertRaisesRegexp(
        ValueError, "skip_connections are enabled but not all cores are "
        "`snt.RNNCore`s, which is not supported"):
      snt.DeepRNN(cells, skip_connections=True)

  def test_non_recurrent_mappings(self):
    insize = 2
    hidden1_size = 4
    hidden2_size = 5
    seq_length = 7
    batch_size = 3

    # As mentioned above, non-recurrent cores are not supported with
    # skip connections. But test that some number of non-recurrent cores
    # is okay (particularly as the last core) without skip connections.
    cores1 = [snt.LSTM(hidden1_size), tf.tanh, snt.Linear(hidden2_size)]
    core1 = snt.DeepRNN(cores1, skip_connections=False)
    core1_h0 = core1.initial_state(batch_size=batch_size)

    cores2 = [snt.LSTM(hidden1_size), snt.Linear(hidden2_size), tf.tanh]
    core2 = snt.DeepRNN(cores2, skip_connections=False)
    core2_h0 = core2.initial_state(batch_size=batch_size)

    xseq = tf.random_normal(shape=[seq_length, batch_size, insize])
    y1, _ = tf.nn.dynamic_rnn(
        core1, xseq, initial_state=core1_h0, time_major=True)
    y2, _ = tf.nn.dynamic_rnn(
        core2, xseq, initial_state=core2_h0, time_major=True)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run([y1, y2])

  def testVariables(self):
    batch_size = 3
    in_size = 2
    hidden1_size = 4
    hidden2_size = 5
    mod_name = "deep_rnn"
    inputs = tf.placeholder(tf.float32, shape=[batch_size, in_size])
    prev_state1 = tf.placeholder(tf.float32, shape=[batch_size, hidden1_size])
    prev_state2 = tf.placeholder(tf.float32, shape=[batch_size, hidden1_size])
    prev_state = (prev_state1, prev_state2)

    cores = [snt.VanillaRNN(name="rnn1", hidden_size=hidden1_size),
             snt.VanillaRNN(name="rnn2", hidden_size=hidden2_size)]
    deep_rnn = snt.DeepRNN(cores, name=mod_name)
    self.assertEqual(deep_rnn.scope_name, mod_name)
    with self.assertRaisesRegexp(snt.Error, "not instantiated yet"):
      deep_rnn.get_variables()
    deep_rnn(inputs, prev_state)

    # No variables now exposed by the DeepRNN.
    self.assertEqual(deep_rnn.get_variables(), ())

    # Have to retrieve the modules from the cores individually.
    deep_rnn_variables = tuple(itertools.chain.from_iterable(
        [c.get_variables() for c in cores]))
    self.assertEqual(len(deep_rnn_variables), 4 * len(cores),
                     "Cores should have %d variables" % (4 * len(cores)))
    for v in deep_rnn_variables:
      self.assertRegexpMatches(
          v.name, "rnn(1|2)/(in_to_hidden|hidden_to_hidden)/(w|b):0")

  @parameterized.Parameters((True, True), (True, False), (False, True),
                            (False, False))
  def testComputation(self, skip_connections, create_initial_state):
    batch_size = 3
    in_size = 2
    hidden1_size = 4
    hidden2_size = 5
    mod_name = "deep_rnn"
    cores = [snt.VanillaRNN(name="rnn1", hidden_size=hidden1_size),
             snt.VanillaRNN(name="rnn2", hidden_size=hidden2_size)]
    deep_rnn = snt.DeepRNN(cores, name=mod_name,
                           skip_connections=skip_connections)
    inputs = tf.placeholder(tf.float32, shape=[batch_size, in_size])
    if create_initial_state:
      prev_state = deep_rnn.initial_state(batch_size, tf.float32)
    else:
      prev_state1 = tf.placeholder(
          tf.float32, shape=[batch_size, hidden1_size])
      prev_state2 = tf.placeholder(
          tf.float32, shape=[batch_size, hidden2_size])
      prev_state = (prev_state1, prev_state2)

    output, next_state = deep_rnn(inputs, prev_state)
    with self.test_session() as sess:
      # With random data, check the DeepRNN calculation matches the manual
      # stacking version.
      input_data = np.random.randn(batch_size, in_size)
      feed_dict = {inputs: input_data}
      if not create_initial_state:
        feed_dict[prev_state1] = np.random.randn(batch_size, hidden1_size)
        feed_dict[prev_state2] = np.random.randn(batch_size, hidden2_size)

      tf.global_variables_initializer().run()

      outputs_value = sess.run([output, next_state[0], next_state[1]],
                               feed_dict=feed_dict)
      output_value, next_state1_value, next_state2_value = outputs_value

      # Build manual computation graph
      output1, next_state1 = cores[0](inputs, prev_state[0])
      if skip_connections:
        input2 = tf.concat([inputs, output1], 1)
      else:
        input2 = output1
      output2, next_state2 = cores[1](input2, prev_state[1])
      if skip_connections:
        manual_output = tf.concat([output1, output2], 1)
      else:
        manual_output = output2
      manual_outputs_value = sess.run([manual_output, next_state1, next_state2],
                                      feed_dict=feed_dict)
    manual_output_value = manual_outputs_value[0]
    manual_next_state1_value = manual_outputs_value[1]
    manual_next_state2_value = manual_outputs_value[2]

    self.assertAllClose(output_value, manual_output_value)
    self.assertAllClose(next_state1_value, manual_next_state1_value)
    self.assertAllClose(next_state2_value, manual_next_state2_value)

  def testNonRecurrentOnly(self):
    batch_size = 3
    in_size = 2
    output1_size = 4
    output2_size = 5

    cores = [snt.Linear(name="linear1", output_size=output1_size),
             snt.Linear(name="linear2", output_size=output2_size)]

    # Build DeepRNN of non-recurrent components.
    deep_rnn = snt.DeepRNN(cores, name="deeprnn", skip_connections=False)
    input_ = tf.placeholder(tf.float32, shape=[batch_size, in_size])
    output, _ = deep_rnn(input_, ())

    # Build manual computation graph.
    output1 = cores[0](input_)
    input2 = output1
    output2 = cores[1](input2)
    manual_output = output2

    with self.test_session() as sess:
      input_data = np.random.randn(batch_size, in_size)
      feed_dict = {input_: input_data}
      tf.global_variables_initializer().run()
      output_value = sess.run([output], feed_dict=feed_dict)
      manual_out_value = sess.run([manual_output], feed_dict=feed_dict)

    self.assertAllClose(output_value, manual_out_value)

  @parameterized.Parameters((False, False), (False, True), (True, False),
                            (True, True))
  def testInitialState(self, trainable, use_custom_initial_value):
    batch_size = 3
    hidden1_size = 4
    hidden2_size = 5
    output1_size = 6
    output2_size = 7

    initializer = None
    if use_custom_initial_value:
      initializer = [tf.constant_initializer(8),
                     tf.constant_initializer(9)]
    # Test that the initial state of a non-recurrent DeepRNN is an empty list.
    non_recurrent_cores = [snt.Linear(output_size=output1_size),
                           snt.Linear(output_size=output2_size)]
    dummy_deep_rnn = snt.DeepRNN(non_recurrent_cores, skip_connections=False)
    dummy_initial_state = dummy_deep_rnn.initial_state(
        batch_size, trainable=trainable)
    self.assertFalse(dummy_initial_state)

    # Test that the initial state of a recurrent DeepRNN is the same as calling
    # all cores' initial_state method.
    cores = [snt.VanillaRNN(hidden_size=hidden1_size),
             snt.VanillaRNN(hidden_size=hidden2_size)]
    deep_rnn = snt.DeepRNN(cores)

    initial_state = deep_rnn.initial_state(batch_size, trainable=trainable,
                                           trainable_initializers=initializer)
    expected_initial_state = []
    for i, core in enumerate(cores):
      with tf.variable_scope("core-%d" % i):
        expected_initializer = None
        if initializer:
          expected_initializer = initializer[i]
        expected_initial_state.append(
            core.initial_state(batch_size, trainable=trainable,
                               trainable_initializers=expected_initializer))

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      initial_state_value = sess.run(initial_state)
      expected_initial_state_value = sess.run(expected_initial_state)

    for expected_value, actual_value in zip(expected_initial_state_value,
                                            initial_state_value):
      self.assertAllEqual(actual_value, expected_value)

  def testInitialStateInModule(self):
    # Check that scopes play nicely with initial states created inside modules.
    batch_size = 6

    def module_build():
      core = snt.DeepRNN([snt.LSTM(4), snt.LSTM(5)])
      initial_state1 = core.initial_state(
          batch_size, dtype=tf.float32, trainable=True)
      initial_state2 = core.initial_state(
          batch_size + 1, dtype=tf.float32, trainable=True)

      return initial_state1, initial_state2

    initial_state_module = snt.Module(module_build)
    initial_state = initial_state_module()

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      initial_state_value = sess.run(initial_state)
      self.assertEqual(initial_state_value[0][0][0].shape, (batch_size, 4))
      self.assertEqual(initial_state_value[1][0][0].shape, (batch_size + 1, 4))
      self.assertEqual(initial_state_value[0][0][1].shape, (batch_size, 4))
      self.assertEqual(initial_state_value[1][0][1].shape, (batch_size + 1, 4))
      self.assertEqual(initial_state_value[0][1][0].shape, (batch_size, 5))
      self.assertEqual(initial_state_value[1][1][0].shape, (batch_size + 1, 5))
      self.assertEqual(initial_state_value[0][1][1].shape, (batch_size, 5))
      self.assertEqual(initial_state_value[1][1][1].shape, (batch_size + 1, 5))

  def testInitialStateNames(self):
    hidden_size_a = 3
    hidden_size_b = 4
    batch_size = 5
    deep_rnn = snt.DeepRNN(
        [snt.LSTM(hidden_size_a, name="a"), snt.LSTM(hidden_size_b, name="b")])
    deep_rnn_state = deep_rnn.initial_state(batch_size, trainable=True)
    self.assertEqual(
        deep_rnn_state[0][0].name,
        "deep_rnn_initial_state/a_initial_state/state_0_tiled:0")
    self.assertEqual(
        deep_rnn_state[0][1].name,
        "deep_rnn_initial_state/a_initial_state/state_1_tiled:0")
    self.assertEqual(
        deep_rnn_state[1][0].name,
        "deep_rnn_initial_state/b_initial_state/state_0_tiled:0")
    self.assertEqual(
        deep_rnn_state[1][1].name,
        "deep_rnn_initial_state/b_initial_state/state_1_tiled:0")

    other_start_state = deep_rnn.initial_state(
        batch_size, trainable=True, name="blah")
    self.assertEqual(other_start_state[0][0].name,
                     "blah/a_initial_state/state_0_tiled:0")
    self.assertEqual(other_start_state[0][1].name,
                     "blah/a_initial_state/state_1_tiled:0")
    self.assertEqual(other_start_state[1][0].name,
                     "blah/b_initial_state/state_0_tiled:0")
    self.assertEqual(other_start_state[1][1].name,
                     "blah/b_initial_state/state_1_tiled:0")

  def testSkipConnectionOptions(self):
    batch_size = 3
    x_seq_shape = [10, batch_size, 2]
    num_hidden = 5
    num_layers = 4
    final_hidden_size = 9
    x_seq = tf.placeholder(shape=x_seq_shape, dtype=tf.float32)
    cores = [snt.LSTM(num_hidden) for _ in xrange(num_layers - 1)]
    final_core = snt.LSTM(final_hidden_size)
    cores += [final_core]
    deep_rnn_core = snt.DeepRNN(cores,
                                skip_connections=True,
                                concat_final_output_if_skip=False)
    initial_state = deep_rnn_core.initial_state(batch_size=batch_size)
    output_seq, _ = tf.nn.dynamic_rnn(deep_rnn_core,
                                      x_seq,
                                      time_major=True,
                                      initial_state=initial_state,
                                      dtype=tf.float32)
    initial_output = output_seq[0]
    feed_dict = {x_seq: np.random.normal(size=x_seq_shape)}
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      initial_output_res = sess.run(initial_output, feed_dict=feed_dict)
    expected_shape = (batch_size, final_hidden_size)
    self.assertSequenceEqual(initial_output_res.shape, expected_shape)

  def testMLPFinalCore(self):
    batch_size = 2
    sequence_length = 3
    input_size = 4
    mlp_last_layer_size = 17
    cores = [
        snt.LSTM(hidden_size=10),
        snt.nets.MLP(output_sizes=[6, 7, mlp_last_layer_size]),
    ]
    deep_rnn = snt.DeepRNN(cores, skip_connections=False)
    input_sequence = tf.constant(
        np.random.randn(sequence_length, batch_size, input_size),
        dtype=tf.float32)
    initial_state = deep_rnn.initial_state(batch_size=batch_size)
    output, unused_final_state = tf.nn.dynamic_rnn(
        deep_rnn, input_sequence,
        initial_state=initial_state,
        time_major=True)
    self.assertEqual(
        output.get_shape(),
        tf.TensorShape([sequence_length, batch_size, mlp_last_layer_size]))

  def testFinalCoreHasNoSizeWarning(self):
    cores = [snt.LSTM(hidden_size=10), snt.Linear(output_size=42), tf.nn.relu]
    rnn = snt.DeepRNN(cores, skip_connections=False)

    with mock.patch.object(tf.logging, "warning") as mocked_logging_warning:
      # This will produce a warning.
      unused_output_size = rnn.output_size
      self.assertTrue(mocked_logging_warning.called)
      first_call_args = mocked_logging_warning.call_args[0]
      self.assertTrue("final core %s does not have the "
                      ".output_size field" in first_call_args[0])
      self.assertEqual(first_call_args[2], 42)

  def testNoSizeButAlreadyConnected(self):
    batch_size = 16
    cores = [snt.LSTM(hidden_size=10), snt.Linear(output_size=42), tf.nn.relu]
    rnn = snt.DeepRNN(cores, skip_connections=False)
    unused_output = rnn(tf.zeros((batch_size, 128)),
                        rnn.initial_state(batch_size=batch_size))

    with mock.patch.object(tf.logging, "warning") as mocked_logging_warning:
      output_size = rnn.output_size
      # Correct size is automatically inferred.
      self.assertEqual(output_size, tf.TensorShape([42]))
      self.assertTrue(mocked_logging_warning.called)
      first_call_args = mocked_logging_warning.call_args[0]
      self.assertTrue("DeepRNN has been connected into the graph, "
                      "so inferred output size" in first_call_args[0])


class ModelRNNTest(tf.test.TestCase):

  def setUp(self):
    self.batch_size = 3
    self.hidden_size = 4
    self.model = snt.Module(name="model", build=tf.identity)
    self.model.output_size = tf.TensorShape(self.hidden_size)

  def testShape(self):
    model_rnn = snt.ModelRNN(self.model)
    inputs = tf.random_normal([self.batch_size, 5])
    prev_state = tf.placeholder(tf.float32,
                                shape=[self.batch_size, self.hidden_size])

    outputs, next_state = model_rnn(inputs, prev_state)
    batch_size_shape = tf.TensorShape(self.batch_size)
    expected_shape = batch_size_shape.concatenate(self.model.output_size)

    self.assertNotEqual(expected_shape, inputs.get_shape())
    self.assertEqual(expected_shape, prev_state.get_shape())
    self.assertEqual(expected_shape, next_state.get_shape())
    self.assertEqual(expected_shape, outputs.get_shape())

  def testComputation(self):
    model_rnn = snt.ModelRNN(self.model)
    inputs = tf.random_normal([self.batch_size, 5])
    prev_state = tf.placeholder(tf.float32,
                                shape=[self.batch_size, self.hidden_size])

    outputs, next_state = model_rnn(inputs, prev_state)

    with self.test_session() as sess:
      prev_state_data = np.random.randn(self.batch_size, self.hidden_size)
      feed_dict = {prev_state: prev_state_data}
      sess.run(tf.global_variables_initializer())

      outputs_value = sess.run([outputs, next_state], feed_dict=feed_dict)
      outputs_value, next_state_value = outputs_value

    self.assertAllClose(prev_state_data, outputs_value)
    self.assertAllClose(outputs_value, next_state_value)

  def testBadArguments(self):
    with self.assertRaises(AttributeError):
      snt.ModelRNN(tf.identity)
    with self.assertRaises(TypeError):
      snt.ModelRNN(np.array([42]))


if __name__ == "__main__":
  tf.test.main()
