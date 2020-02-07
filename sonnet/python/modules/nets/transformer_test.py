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
"""Tests for sonnet transformer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import sonnet as snt
import tensorflow.compat.v1 as tf


class TransformerTowerTest(tf.test.TestCase):

  def test_forward(self):
    batch_size = 8
    window_size = 15
    value_size = 6
    num_heads = 16
    input_size = num_heads * value_size
    output_size = 128
    inputs = tf.random_normal([batch_size, window_size, input_size])
    transformer = snt.nets.TransformerTower(
        value_size=value_size,
        num_heads=num_heads,
        num_layers=3,
        causal=False,
        shared_attention=False,
        output_size=output_size,
        mlp_hidden_sizes=tuple([64]))
    output, _ = transformer(inputs)
    self.assertAllEqual(output.get_shape().as_list(),
                        [batch_size, window_size, output_size])

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(output)

  def test_invalid_input(self):
    batch_size = 8
    window_size = 15
    value_size = 6
    num_heads = 16
    # invalid input size because it is odd
    input_size = num_heads * value_size + 1
    output_size = 128
    invalid_inputs = tf.random_normal([batch_size, window_size, input_size])
    transformer = snt.nets.TransformerTower(
        value_size=value_size,
        num_heads=num_heads,
        num_layers=3,
        causal=False,
        shared_attention=False,
        output_size=output_size,
        use_relative_positions=False,
        mlp_hidden_sizes=tuple([64]))

    with self.assertRaises(ValueError):
      transformer(invalid_inputs)


class TransformerXLTest(tf.test.TestCase):

  def check_memory_gradients(self,
                             state,
                             output,
                             session,
                             start=0,
                             end=-1,
                             zero=True):
    """Checks masking via norm of state gradient with respect to output.

    Args:
      state: transformer.AttentionCoreState.
      output: tensor of model outputs.
      session: tensorflow session.
      start: inspect gradients from [start:] slots in memory.
      end: inspect gradients up to [:end] slots in memory.
      zero: if true, checks equal to zero, otherwise checks greater than zero.
    """
    for state_i in state:
      grad_i = tf.gradients(output, state_i)[0][:, start:end]
      grad_norm_i = tf.reduce_sum(tf.square(grad_i))

      grad_norm_np = session.run(grad_norm_i)
      if zero:
        self.assertEqual(grad_norm_np, 0)
      else:
        self.assertGreater(grad_norm_np, 0)

  def test_forward(self):
    batch_size = 8
    window_size = 15
    input_size = 16
    output_size = 128
    num_layers = 3
    key_size = 4
    value_size = 6
    num_heads = 16
    memory_size = 48
    inputs = tf.random_normal([batch_size, window_size, input_size])
    core_config = {
        'value_size': value_size,
        'key_size': key_size,
        'num_heads': num_heads,
        'num_layers': num_layers,
        'causal': True,
        'shared_attention': False,
        'output_size': output_size,
        'mlp_hidden_sizes': tuple([64]),
    }
    transformer_xl = snt.nets.transformer.TransformerXL(
        core_config=core_config,
        memory_size=memory_size,
        chunk_size=window_size,
    )
    initial_state = transformer_xl.initial_state(batch_size)
    output, next_state = transformer_xl(inputs, initial_state)
    output2, final_state = transformer_xl(inputs, next_state)
    self.assertAllEqual(output.get_shape().as_list(),
                        [batch_size, window_size, output_size])
    self.assertEqual(len(next_state), num_layers)

    def check_state_size(state_list):
      for i in range(num_layers):
        state = state_list[i]
        self.assertAllEqual(state.get_shape().as_list(),
                            [batch_size, memory_size, value_size * num_heads])

    check_state_size(next_state)
    check_state_size(final_state)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(output)
      sess.run(next_state)
      sess.run(output2)
      sess.run(final_state)

  def test_mask_op(self):
    logits = tf.zeros(shape=(1, 1, 5, 5))
    masked_logits = logits + snt.nets.transformer.future_mask(
        chunk_size=5, dtype=logits.dtype)
    weights = tf.nn.softmax(logits)
    masked_weights = tf.nn.softmax(masked_logits)
    with self.test_session() as sess:
      weights_v, masked_weights_v = sess.run([weights, masked_weights])

    expected_weights_v = np.array([
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [0.2, 0.2, 0.2, 0.2, 0.2],
    ]).reshape([1, 1, 5, 5])
    self.assertAllClose(weights_v, expected_weights_v)

    expected_masked_weights_v = np.array(
        [[1. / 1, 0.00, 0.00, 0.00, 0.00], [1. / 2, 1. / 2, 0.00, 0.00, 0.00],
         [1. / 3, 1. / 3, 1. / 3, 0.00, 0.00],
         [1. / 4, 1. / 4, 1. / 4, 1. / 4, 0.00],
         [1. / 5, 1. / 5, 1. / 5, 1. / 5, 1. / 5]]).reshape([1, 1, 5, 5])
    self.assertAllClose(masked_weights_v, expected_masked_weights_v)

  def test_masking_no_memory(self):
    """Checks that masking disallows information flow from future to present."""

    batch_size = 1
    value_size = 6
    num_heads = 16
    hidden_size = value_size * num_heads
    seq_length = 3
    decoder_input = tf.random_normal([batch_size, seq_length, hidden_size])
    transformer = snt.nets.TransformerTower(
        value_size=value_size,
        num_heads=num_heads,
        num_layers=3,
        causal=True,
        shared_attention=False,
        output_size=hidden_size,
        mlp_hidden_sizes=tuple([64]))
    decoder_output, _ = transformer(decoder_input)

    # For each time step of the output sequence, compute the
    # derivative with respect to each component of whole input tensor.
    # Sum over input and output channels.
    gradients = []
    for time_idx in range(seq_length):
      tf.logging.info('Creating gradient ops for time %d/%d.' %
                      (time_idx, seq_length))

      time_gradients = tf.gradients(
          decoder_output[0, time_idx],  # Sums over output channels
          decoder_input)[0]
      gradients.append(time_gradients)
    gradients = tf.stack(gradients, 0)
    tf.logging.info('Done creating gradient ops.')

    with self.test_session() as session:
      tf.global_variables_initializer().run()
      tf.logging.info('Fetching gradient ops.')
      output_v, grad_v = session.run([decoder_output, gradients])
      tf.logging.info('Done fetching gradient ops.')

    # Pick out the subset of derivatives which should be zero
    # and test for exact equality with zero.
    time_grad_v = np.sum(  # Sum over input channels.
        grad_v[:, 0, :], axis=2)
    grad_triu = time_grad_v[np.triu_indices(seq_length, k=1)]
    self.assertAllEqual(grad_triu, np.zeros_like(grad_triu))

    # Make sure there are no nans in the output.
    self.assertTrue(np.all(np.logical_not(np.isnan(output_v))))

  def test_no_dropout_during_eval(self):
    """Checks that dropout is only applied during training, and not eval."""
    batch_size = 2
    sequence_length = 10
    memory_size = 48
    core_config = {
        'key_size': 3,
        'value_size': 4,
        'num_heads': 5,
        'num_layers': 2,
        'dropout_rate': 0.5,
    }
    inputs = tf.ones([batch_size, sequence_length, 16], dtype=tf.float32)
    transformer_xl = snt.nets.transformer.TransformerXL(
        core_config, memory_size, chunk_size=sequence_length)
    initial_state = transformer_xl.initial_state(batch_size, dtype=inputs.dtype)
    eval_output, _ = transformer_xl(inputs, initial_state, is_training=False)
    train_output, _ = transformer_xl(inputs, initial_state, is_training=True)
    with self.test_session() as session:
      tf.global_variables_initializer().run()
      # Ensures dropout is being applied during training (output changes).
      train_out_sum = session.run(tf.reduce_sum(train_output))
      train_out_sum2 = session.run(tf.reduce_sum(train_output))
      self.assertNotAlmostEqual(train_out_sum, train_out_sum2)

      # Ensures dropout is not being applied during eval (output is same).
      eval_out_sum = session.run(tf.reduce_sum(eval_output))
      eval_out_sum2 = session.run(tf.reduce_sum(eval_output))
      self.assertAlmostEqual(eval_out_sum, eval_out_sum2)

  def test_zero_chunk_size(self):
    """Tests a chunk size of 0 corresponds to a regular RNN core."""
    batch_size = 2
    core_config = {
        'key_size': 3,
        'value_size': 4,
        'num_heads': 5,
        'num_layers': 2,
        'dropout_rate': 0.5,
    }
    inputs = tf.ones([10, batch_size, 16], dtype=tf.float32)
    transformer_xl = snt.nets.transformer.TransformerXL(
        core_config, memory_size=8, chunk_size=0)
    initial_state = transformer_xl.initial_state(batch_size)
    output, final_state = tf.nn.dynamic_rnn(
        transformer_xl, inputs, time_major=True, initial_state=initial_state)
    with self.test_session() as session:
      tf.global_variables_initializer().run()
      session.run([output, final_state])

  def test_zero_memory_size(self):
    """Tests a memory size of 0 corresponds to a regular RNN core."""
    batch_size = 2
    memory_size = 0
    sequence_length = 10
    core_config = {
        'key_size': 3,
        'value_size': 4,
        'num_heads': 5,
        'num_layers': 2,
        'dropout_rate': 0.5,
    }
    inputs = tf.ones([batch_size, sequence_length, 16], dtype=tf.float32)
    transformer_xl = snt.nets.transformer.TransformerXL(
        core_config, memory_size=memory_size, chunk_size=sequence_length)
    initial_state = transformer_xl.initial_state(batch_size)
    output, final_state = transformer_xl(inputs, initial_state)
    with self.test_session() as session:
      tf.global_variables_initializer().run()
      session.run([output, final_state])

  def test_dynamic_batch_size(self):
    """Tests operation with changing batch size."""
    memory_size = 0
    sequence_length = 10
    core_config = {
        'key_size': 3,
        'value_size': 4,
        'num_heads': 5,
        'num_layers': 2,
        'dropout_rate': 0.5,
    }
    inputs = tf.placeholder(tf.float32, shape=(None, sequence_length, 16))
    batch_size = tf.shape(inputs)[0]
    transformer_xl = snt.nets.transformer.TransformerXL(
        core_config, memory_size=memory_size, chunk_size=sequence_length)
    initial_state = transformer_xl.initial_state(batch_size)
    output, final_state = transformer_xl(inputs, initial_state)
    with self.test_session() as session:
      tf.global_variables_initializer().run()
      batch_size_1 = 2
      final_output_1, _ = session.run(
          [output, final_state],
          feed_dict={inputs: np.ones([batch_size_1, sequence_length, 16])})
      self.assertAllEqual(final_output_1.shape[0], batch_size_1)

      batch_size_2 = 4
      final_output_2, _ = session.run(
          [output, final_state],
          feed_dict={inputs: np.ones([batch_size_2, sequence_length, 16])})
      self.assertAllEqual(final_output_2.shape[0], batch_size_2)


class CompressiveTransformerTest(tf.test.TestCase):

  def test_forward(self):
    batch_size = 8
    window_size = 18
    input_size = 16
    output_size = 128
    num_layers = 3
    key_size = 4
    value_size = 6
    num_heads = 16
    em_memory_size = 18
    cm_memory_size = 7
    inputs = tf.random_normal([batch_size, window_size, input_size])
    core_config = {
        'value_size': value_size,
        'key_size': key_size,
        'num_heads': num_heads,
        'num_layers': num_layers,
        'causal': True,
        'shared_attention': False,
        'output_size': output_size,
        'mlp_hidden_sizes': tuple([64]),
    }
    compressive_transformer = snt.nets.CompressiveTransformer(
        core_config=core_config,
        episodic_memory_size=em_memory_size,
        compressed_memory_size=cm_memory_size,
        chunk_size=window_size,
    )
    initial_state = compressive_transformer.initial_state(batch_size)
    output, next_state = compressive_transformer(inputs, initial_state)
    output2, final_state = compressive_transformer(inputs, next_state)
    compression_loss = tf.get_collection('auxiliary_losses')
    self.assertAllEqual(output.get_shape().as_list(),
                        [batch_size, window_size, output_size])
    self.assertEqual(len(next_state), num_layers)
    self.assertEqual(len(next_state[0]), 3)  # index, cm, em

    def check_state_size(state_list):
      for state in state_list:
        self.assertAllEqual(
            state.episodic_memory.get_shape().as_list(),
            [batch_size, em_memory_size, value_size * num_heads])
        self.assertAllEqual(
            state.compressed_memory.get_shape().as_list(),
            [batch_size, cm_memory_size, value_size * num_heads])

    check_state_size(next_state)
    check_state_size(final_state)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(output)
      sess.run(next_state)
      sess.run(output2)
      sess.run(final_state)
      compression_loss_np = sess.run(compression_loss)
    # Compression loss is zero because em and cm are zero.
    self.assertEqual(compression_loss_np[0], 0)
    # Compression loss is > 0 because em is populated.
    self.assertGreater(compression_loss_np[1], 0)


if __name__ == '__main__':
  tf.test.main()
