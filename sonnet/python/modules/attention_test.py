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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
import sonnet as snt
from sonnet.testing import parameterized
import tensorflow as tf


class ConstantZero(snt.AbstractModule):
  """A module that always outputs zero for each example."""

  def __init__(self, output_rank=2, name="constant_zero"):
    """Initialize ConstantZero module.

    Args:
      output_rank: int. Rank of value returned by build(). The default value (2)
        imitates the output of the Linear module.
      name: string. Name of module.
    """
    super(ConstantZero, self).__init__(name=name)
    self._output_rank = output_rank

  def _build(self, inputs):
    """Attach ConstantZero module to graph.

    Args:
      inputs: [batch_size, input_size]-shaped Tensor of dtype float32.

    Returns:
      A Tensor with rank output_rank where the first dimension has length
        batch_size and all others have length 1.
    """
    # A module like Linear would require the final dimension to be known in
    # order to construct weights.
    assert inputs.get_shape().as_list()[-1] is not None
    batch_size = tf.shape(inputs)[0]
    result_shape = [batch_size] + [1] * (self._output_rank - 1)
    return tf.zeros(result_shape, dtype=inputs.dtype)


class AttentiveReadTest(tf.test.TestCase, parameterized.ParameterizedTestCase):

  def setUp(self):
    super(AttentiveReadTest, self).setUp()

    self._batch_size = 3
    self._memory_size = 4
    self._memory_word_size = 1
    self._query_word_size = 2
    self._memory = tf.reshape(
        tf.cast(tf.range(0, 3 * 4 * 1), dtype=tf.float32), shape=[3, 4, 1])
    self._query = tf.reshape(
        tf.cast(tf.range(0, 3 * 2), dtype=tf.float32), shape=[3, 2])
    self._memory_mask = tf.convert_to_tensor(
        [
            [True, True, True, True],
            [True, True, True, False],
            [True, True, False, False],
        ],
        dtype=tf.bool)
    self._attention_logit_mod = ConstantZero()
    self._attention_mod = snt.AttentiveRead(self._attention_logit_mod)

  def testShape(self):
    # Shape should be inferred if it's known at graph construction time.
    attention_output = self._attention_mod(self._memory, self._query)
    x = attention_output.read
    self.assertTrue(x.get_shape().is_compatible_with(
        [self._batch_size, self._memory_word_size]))
    self.assertEqual(x.dtype, tf.float32)

  def testComputation(self):
    # Since all attention weight logits are zero, all memory slots get an equal
    # weight. Thus, the expected attentive read should return the average of
    # all memory slots for each example.
    attention_output = self._attention_mod(self._memory, self._query)
    x = attention_output.read
    with self.test_session() as sess:
      x_ = sess.run(x)
      self.assertAllClose(x_, [[1.5], [5.5], [9.5]])

  def testMemoryMask(self):
    # Ignore some time steps.
    attention_output = self._attention_mod(
        self._memory, self._query, memory_mask=self._memory_mask)
    x = attention_output.read
    with self.test_session() as sess:
      x_ = sess.run(x)
      self.assertAllClose(x_, [[1.5], [5.0], [8.5]])

  def testMemoryMaskWithNonuniformLogits(self):
    memory = np.random.randn(2, 3, 10)
    logits = np.array([[-1, 1, 0], [-1, 1, 0]])
    mask = np.array([[True, True, True], [True, True, False]])

    # Calculate expected output.
    expected_weights = np.exp(logits)
    expected_weights[1, 2] = 0
    expected_weights /= np.sum(expected_weights, axis=1, keepdims=True)
    expected_output = np.matmul(expected_weights[:, np.newaxis, :],
                                memory)[:, 0]

    # Run attention model.
    attention = snt.AttentiveRead(
        lambda _: tf.constant(logits.reshape([6, 1]), dtype=tf.float32))
    attention_output = attention(
        memory=tf.constant(memory, dtype=tf.float32),
        query=tf.constant(np.zeros([2, 5]), dtype=tf.float32),
        memory_mask=tf.constant(mask))
    with self.test_session() as sess:
      actual = sess.run(attention_output)

    # Check output.
    self.assertAllClose(actual.read, expected_output)
    self.assertAllClose(actual.weights, expected_weights)
    # The actual logit for the masked value should be tiny. First check without.
    masked_actual_weight_logits = np.array(actual.weight_logits, copy=True)
    masked_actual_weight_logits[1, 2] = logits[1, 2]
    self.assertAllClose(masked_actual_weight_logits, logits)
    self.assertLess(actual.weight_logits[1, 2], -1e35)

  def testUndefinedWordSizes(self):
    # memory_word_size must be defined.
    memory = tf.placeholder(
        dtype=tf.float32, shape=[self._batch_size, self._memory_size, None])
    with self.assertRaises(snt.UnderspecifiedError):
      self._attention_mod(memory, self._query)

    # query_word_size must be defined.
    query = tf.placeholder(dtype=tf.float32, shape=[self._batch_size, None])
    with self.assertRaises(snt.UnderspecifiedError):
      self._attention_mod(self._memory, query)

  def testMemoryShape(self):
    # memory must have rank 3.
    memory = tf.placeholder(
        dtype=tf.float32, shape=[self._batch_size, self._memory_size])
    with self.assertRaises(snt.IncompatibleShapeError):
      self._attention_mod(memory, self._query)

  def testQueryShape(self):
    # query must have rank 2.
    query = tf.placeholder(
        dtype=tf.float32, shape=[self._batch_size, self._query_word_size, 1])
    with self.assertRaises(snt.IncompatibleShapeError):
      self._attention_mod(self._memory, query)

  def testMemoryMaskShape(self):
    # memory_mask must have rank 2.
    memory_mask = tf.placeholder(
        dtype=tf.bool, shape=[self._batch_size, self._memory_size, 1])
    with self.assertRaises(snt.IncompatibleShapeError):
      self._attention_mod(self._memory, self._query, memory_mask=memory_mask)

  @parameterized.Parameters(1, 3)
  def testAttentionLogitsModuleShape(self, output_rank):
    # attention_logit_mod must produce a rank 2 Tensor.
    attention_mod = snt.AttentiveRead(ConstantZero(output_rank=output_rank))
    with self.assertRaises(snt.IncompatibleShapeError):
      attention_mod(self._memory, self._query)

  def testNoMemorySlotsLeft(self):
    # Every example must have at least one unmasked memory slot for attention
    # to work.
    memory_mask = tf.convert_to_tensor(
        [
            [True, True, True, True],
            [True, True, True, False],
            [False, False, False, False],
        ],
        dtype=tf.bool)
    attention_output = self._attention_mod(
        self._memory, self._query, memory_mask=memory_mask)
    x = attention_output.read
    with self.test_session() as sess:
      with self.assertRaises(tf.errors.InvalidArgumentError):
        sess.run(x)

  def testInvalidBatchSize(self):
    # Both memory and query need to agree on batch_size.
    memory = tf.placeholder(shape=[None, 1, 1], dtype=tf.float32)
    query = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    attention_output = self._attention_mod(memory, query)
    x = attention_output.read
    with self.test_session() as sess:
      with self.assertRaises(tf.errors.InvalidArgumentError):
        feed_dict = {
            memory: np.zeros([1, 1, 1], dtype=np.float32),
            query: np.zeros([2, 1], dtype=np.float32)
        }
        sess.run(x, feed_dict=feed_dict)

  @parameterized.Parameters({
      "module_cstr": snt.Linear,
      "module_kwargs": {
          "output_size": 1
      }
  }, {"module_cstr": snt.nets.MLP,
      "module_kwargs": {
          "output_sizes": [1]
      }})
  def testWorksWithCommonModules(self, module_cstr, module_kwargs):
    # In the academic literature, attentive reads are most commonly implemented
    # with Linear or MLP modules. This integration test ensures that
    # AttentiveRead works safely with these.

    attention_logit_mod = module_cstr(**module_kwargs)
    attention_mod = snt.AttentiveRead(attention_logit_mod)
    x = attention_mod(self._memory, self._query)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(x)

  def testAttentionWeightLogitsShape(self):
    # Expected to be [batch_size, memory_size].
    x = self._attention_mod(self._memory, self._query).weight_logits
    self.assertTrue(x.get_shape().is_compatible_with(
        [self._batch_size, self._memory_size]))
    self.assertEqual(x.dtype, tf.float32)

  def testWeightsIsSoftmaxOfLogits(self):
    attention_output = self._attention_mod(self._memory, self._query)
    softmax_of_weight_logits = tf.nn.softmax(attention_output.weight_logits)
    with self.test_session() as sess:
      expected, obtained = sess.run([attention_output.weights,
                                     softmax_of_weight_logits])
    self.assertAllClose(expected, obtained)


if __name__ == "__main__":
  tf.test.main()
