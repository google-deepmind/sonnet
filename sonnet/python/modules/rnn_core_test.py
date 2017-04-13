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

"""Tests for Recurrent cores in sonnet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import mock
import numpy as np
import sonnet as snt
from sonnet.testing import parameterized
import tensorflow as tf

from tensorflow.python.util import nest

BATCH_SIZE = 5
MASK_TUPLE = (True, (False, True))

_state_size_tuple = (3, (4, 5))
_state_size_element = 6


# Use patch to instantiate RNNCore
@mock.patch.multiple(snt.RNNCore, __abstractmethods__=set())
class RNNCoreTest(tf.test.TestCase, parameterized.ParameterizedTestCase):

  @parameterized.Parameters(
      (False, False, _state_size_tuple),
      (False, True, _state_size_tuple),
      (True, False, _state_size_tuple),
      (True, True, _state_size_tuple),
      (False, False, _state_size_element),
      (False, True, _state_size_element),
      (True, False, _state_size_element),
      (True, True, _state_size_element))
  def testInitialStateTuple(self, trainable, use_custom_initial_value,
                            state_size):
    batch_size = 6

    # Set the attribute to the class since it we can't set properties of
    # abstract classes
    snt.RNNCore.state_size = state_size
    flat_state_size = nest.flatten(state_size)
    core = snt.RNNCore(name="dummy_core")
    if use_custom_initial_value:
      flat_initializer = [tf.constant_initializer(2)] * len(flat_state_size)
      trainable_initializers = nest.pack_sequence_as(
          structure=state_size, flat_sequence=flat_initializer)
    else:
      trainable_initializers = None
    initial_state = core.initial_state(
        batch_size, dtype=tf.float32, trainable=trainable,
        trainable_initializers=trainable_initializers)

    nest.assert_same_structure(initial_state, state_size)
    flat_initial_state = nest.flatten(initial_state)

    for state, size in zip(flat_initial_state, flat_state_size):
      self.assertEqual(state.get_shape(), [batch_size, size])

    with self.test_session() as sess:
      tf.global_variables_initializer().run()
      flat_initial_state_value = sess.run(flat_initial_state)
      for value, size in zip(flat_initial_state_value, flat_state_size):
        expected_initial_state = np.empty([batch_size, size])
        if not trainable:
          expected_initial_state.fill(0)
        elif use_custom_initial_value:
          expected_initial_state.fill(2)
        else:
          value_row = value[0]
          expected_initial_state = np.tile(value_row, (batch_size, 1))
        self.assertAllClose(value, expected_initial_state)

  @parameterized.Parameters(
      (False, _state_size_tuple),
      (True, _state_size_tuple),
      (False, _state_size_element),
      (True, _state_size_element))
  def testRegularizers(self, trainable, state_size):
    batch_size = 6

    # Set the attribute to the class since it we can't set properties of
    # abstract classes
    snt.RNNCore.state_size = state_size
    flat_state_size = nest.flatten(state_size)
    core = snt.RNNCore(name="dummy_core")
    flat_regularizer = ([tf.contrib.layers.l1_regularizer(scale=0.5)] *
                        len(flat_state_size))
    trainable_regularizers = nest.pack_sequence_as(
        structure=state_size, flat_sequence=flat_regularizer)

    core.initial_state(batch_size, dtype=tf.float32, trainable=trainable,
                       trainable_regularizers=trainable_regularizers)

    graph_regularizers = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    if not trainable:
      self.assertFalse(graph_regularizers)
    else:
      for i in range(len(flat_state_size)):
        self.assertRegexpMatches(
            graph_regularizers[i].name, ".*l1_regularizer.*")


class TrainableInitialState(tf.test.TestCase,
                            parameterized.ParameterizedTestCase):

  @parameterized.Parameters((True, MASK_TUPLE), (True, None), (False, False),
                            (False, None))
  def testInitialStateComputation(self, tuple_state, mask):
    if tuple_state:
      initial_state = (tf.fill([BATCH_SIZE, 6], 2),
                       (tf.fill([BATCH_SIZE, 7], 3),
                        tf.fill([BATCH_SIZE, 8], 4)))
    else:
      initial_state = tf.fill([BATCH_SIZE, 9], 10)

    trainable_state_module = snt.TrainableInitialState(initial_state, mask=mask)
    trainable_state = trainable_state_module()
    nest.assert_same_structure(initial_state, trainable_state)
    flat_initial_state = nest.flatten(initial_state)
    flat_trainable_state = nest.flatten(trainable_state)
    if mask is not None:
      flat_mask = nest.flatten(mask)
    else:
      flat_mask = (True,) * len(flat_initial_state)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())

      # Check all variables are initialized correctly and return a state that
      # has the same as it is provided.
      for trainable_state, initial_state in zip(flat_trainable_state,
                                                flat_initial_state):
        self.assertAllEqual(sess.run(trainable_state), sess.run(initial_state))

      # Change the value of all the trainable variables to ones.
      for variable in tf.trainable_variables():
        sess.run(tf.assign(variable, tf.ones_like(variable)))

      # Check that the values of the initial_states have changed if and only if
      # they are trainable.
      for trainable_state, initial_state, mask in zip(flat_trainable_state,
                                                      flat_initial_state,
                                                      flat_mask):
        trainable_state_value = sess.run(trainable_state)
        initial_state_value = sess.run(initial_state)
        if mask:
          expected_value = np.ones_like(initial_state_value)
        else:
          expected_value = initial_state_value

        self.assertAllEqual(trainable_state_value, expected_value)

  def testBadArguments(self):
    initial_state = (tf.random_normal([BATCH_SIZE, 6]),
                     (tf.random_normal([BATCH_SIZE, 7]),
                      tf.random_normal([BATCH_SIZE, 8])))
    with self.assertRaises(TypeError):
      snt.TrainableInitialState(initial_state, mask=(True, (False, "foo")))

    snt.TrainableInitialState(initial_state, mask=(True, (False, True)))()
    with self.test_session() as sess:
      with self.assertRaises(tf.errors.InvalidArgumentError):
        # Check that the class checks that the elements of initial_state have
        # identical rows.
        sess.run(tf.global_variables_initializer())


if __name__ == "__main__":
  tf.test.main()
