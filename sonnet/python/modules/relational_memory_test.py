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

"""Tests for sonnet.python.modules.relational_memory."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from absl.testing import parameterized
import numpy as np
from sonnet.python.modules import relational_memory
import tensorflow as tf


class RelationalMemoryTest(parameterized.TestCase, tf.test.TestCase):

  def testStateSizeOutputSize(self):
    """Checks for correct `state_size` and `output_size` return values."""
    mem_slots = 4
    head_size = 32
    mem = relational_memory.RelationalMemory(mem_slots, head_size)

    self.assertItemsEqual([mem._mem_slots, mem._mem_size],
                          mem.state_size.as_list())
    self.assertItemsEqual([mem._mem_slots * mem._mem_size],
                          mem.output_size.as_list())

  @parameterized.named_parameters(
      ("PreserveMatrixInput", True), ("DontPreserveMatrixInput", False)
  )
  def testOutputStateShapes(self, treat_input_as_matrix):
    """Checks the shapes of RelationalMemory output and state."""
    mem_slots = 4
    head_size = 32
    num_heads = 2
    batch_size = 5

    input_shape = (batch_size, 3, 3)
    mem = relational_memory.RelationalMemory(mem_slots, head_size, num_heads)
    inputs = tf.placeholder(tf.float32, input_shape)
    init_state = mem.initial_state(batch_size)
    out = mem(inputs, init_state, treat_input_as_matrix=treat_input_as_matrix)

    with self.test_session() as session:
      tf.global_variables_initializer().run()
      new_out, new_memory = session.run(
          out, feed_dict={inputs: np.zeros(input_shape)}
      )
    self.assertAllEqual(init_state.get_shape().as_list(), new_memory.shape)
    self.assertAllEqual(new_out.shape,
                        [batch_size, mem_slots * head_size * num_heads])

  # Check different combinations of mem_slots and mem_size
  # (ie, head_size * num_heads) size to make sure init_state construction
  # works correctly.
  @parameterized.parameters(*zip(
      (2, 4, 8, 16), (2, 4, 8, 16), (1, 2, 3, 4)
  ))
  def testRecurrence(self, mem_slots, head_size, num_heads):
    """Checks if you can run the relational memory for 2 steps."""

    batch_size = 5
    num_blocks = 5

    input_shape = [batch_size, 3, 1]
    mem = relational_memory.RelationalMemory(mem_slots, head_size, num_heads,
                                             num_blocks=num_blocks)
    inputs = tf.placeholder(tf.float32, input_shape)

    hidden_0 = mem.initial_state(batch_size)
    _, hidden_1 = mem(inputs, hidden_0)
    _, hidden_2 = mem(inputs, hidden_1)

    with self.test_session() as session:
      tf.global_variables_initializer().run()
      results = session.run(
          {"hidden_2": hidden_2, "hidden_1": hidden_1},
          feed_dict={inputs: np.zeros(input_shape)}
      )
    self.assertAllEqual(results["hidden_1"].shape, results["hidden_2"].shape)

  def testBadInputs(self):
    """Test that verifies errors are thrown for bad input arguments."""

    mem_slots = 4
    head_size = 32

    with self.assertRaisesRegexp(ValueError, "num_blocks must be >= 1"):
      relational_memory.RelationalMemory(mem_slots, head_size, num_blocks=0)

    with self.assertRaisesRegexp(ValueError,
                                 "attention_mlp_layers must be >= 1"):
      relational_memory.RelationalMemory(mem_slots, head_size,
                                         attention_mlp_layers=0)

    with self.assertRaisesRegexp(ValueError, "gate_style must be one of"):
      relational_memory.RelationalMemory(mem_slots, head_size,
                                         gate_style="bad_gate")

  @parameterized.named_parameters(
      ("GateStyleUnit", "unit"), ("GateStyleMemory", "memory")
  )
  def testGateShapes(self, gate_style):
    """Checks the shapes of RelationalMemory gates."""
    mem_slots = 4
    head_size = 32
    num_heads = 4
    batch_size = 4
    input_shape = (batch_size, 3, 3)

    mem = relational_memory.RelationalMemory(mem_slots, head_size, num_heads,
                                             gate_style=gate_style)

    inputs = tf.placeholder(tf.float32, input_shape)
    init_state = mem.initial_state(batch_size)
    mem(inputs, init_state)

    gate_size = mem._calculate_gate_size()
    expected_size = [batch_size, num_heads, gate_size]

    self.assertEqual(mem.input_gate.get_shape().as_list(), expected_size)
    self.assertEqual(mem.forget_gate.get_shape().as_list(), expected_size)

  def testMemoryUpdating(self):
    """Checks if memory is updating correctly."""
    mem_slots = 2
    head_size = 32
    num_heads = 4
    batch_size = 5
    input_shape = (batch_size, 3, 3)
    mem = relational_memory.RelationalMemory(mem_slots, head_size, num_heads,
                                             gate_style=None)
    inputs = tf.placeholder(tf.float32, input_shape)

    memory_0 = mem.initial_state(batch_size)
    _, memory_1 = mem(inputs, memory_0)

    with self.test_session() as session:
      tf.global_variables_initializer().run()
      results = session.run(
          {"memory_1": memory_1, "memory_0": memory_0},
          feed_dict={inputs: np.zeros(input_shape)})

    self.assertTrue(np.any(np.not_equal(results["memory_0"],
                                        results["memory_1"])))

  @parameterized.named_parameters(
      ("GateStyleUnit", "unit"), ("GateStyleMemory", "memory")
  )
  def testInputErasureWorking(self, gate_style):
    """Checks if gating is working by ignoring the input."""
    mem_slots = 2
    head_size = 32
    num_heads = 2
    batch_size = 5
    input_shape = (batch_size, 3, 3)
    mem = relational_memory.RelationalMemory(mem_slots, head_size, num_heads,
                                             forget_bias=float("+inf"),
                                             input_bias=float("-inf"),
                                             gate_style=gate_style)
    inputs = tf.placeholder(tf.float32, input_shape)

    memory_0 = mem.initial_state(batch_size)
    _, memory_1 = mem(inputs, memory_0)

    with self.test_session() as session:
      tf.global_variables_initializer().run()
      results = session.run(
          {"memory_1": memory_1, "memory_0": memory_0},
          feed_dict={inputs: np.ones(input_shape)})
    self.assertAllEqual(results["memory_0"], results["memory_1"])

  @parameterized.named_parameters(
      ("GateStyleUnit", "unit"), ("GateStyleMemory", "memory")
  )
  def testDifferingKeyHeadSizes(self, gate_style):
    """Checks if arbitrary key sizes are still supported."""
    mem_slots = 2
    head_size = 32
    num_heads = 2
    key_size = 128
    batch_size = 5

    input_shape = (batch_size, 3, 3)
    mem = relational_memory.RelationalMemory(mem_slots, head_size, num_heads,
                                             gate_style=gate_style,
                                             key_size=key_size)
    self.assertNotEqual(key_size, mem._head_size)
    inputs = tf.placeholder(tf.float32, input_shape)

    memory_0 = mem.initial_state(batch_size)
    _, memory_1 = mem(inputs, memory_0)

    with self.test_session() as session:
      tf.global_variables_initializer().run()
      results = session.run(
          {"memory_1": memory_1, "memory_0": memory_0},
          feed_dict={inputs: np.ones(input_shape)})

    self.assertTrue(np.any(np.not_equal(results["memory_0"],
                                        results["memory_1"])))

if __name__ == "__main__":
  tf.test.main()
