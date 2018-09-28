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

"""Example script to train the Relational Memory Core.

This is a reduced size version of the "Learning To Execute" (LTE) task defined
in:

  https://arxiv.org/abs/1806.01822
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
# Dependency imports

from absl import flags
import six

import sonnet as snt
from sonnet.examples import learn_to_execute
import tensorflow as tf


FLAGS = flags.FLAGS

flags.DEFINE_float("learning_rate", 1e-4, "Initial learning rate.")
flags.DEFINE_float("min_learning_rate", 8e-5, "Minimum learning rate.")
flags.DEFINE_integer("batch_size", 1600, "Batch size.")
flags.DEFINE_integer("head_size", 2048, "Total memory size for the RMC.")
flags.DEFINE_integer("num_heads", 1, "Attention heads for RMC.")
flags.DEFINE_integer("num_mems", 4, "Number of memories for RMC.")
flags.DEFINE_integer("num_blocks", 1, "Number of attention blocks for RMC.")
flags.DEFINE_string("gate_style", "unit", "Gating style for RMC.")
flags.DEFINE_integer("max_length", 5, "LTE max literal length.")
flags.DEFINE_integer("max_nest", 2, "LTE max nesting level.")
flags.DEFINE_integer("epochs", 1000000, "Total training epochs.")
flags.DEFINE_integer("log_stride", 500, "Iterations between reports.")


class SequenceModel(snt.AbstractModule):
  """Seq2Seq Model to process LTE sequence batches."""

  def __init__(
      self,
      core,
      target_size,
      final_mlp,
      name="sequence_model"):
    super(SequenceModel, self).__init__(name=name)
    self._core = core
    self._target_size = target_size
    self._final_mlp = final_mlp

  def _build(
      self, inputs, targets, input_sequence_length, output_sequence_length):
    """Dynamic unroll across input objects.

    Args:
      inputs: tensor (input_sequence_length x batch x feature_size). Encoder
          sequence.
      targets: tensor (output_sequence_length x batch x feature_size). Decoder
          sequence.
      input_sequence_length: tensor (batch). Size of each batched input
          sequence.
      output_sequence_length: tensor (batch). Size of each batched target
          sequence.

    Returns:
      Tensor (batch x num_objects); logits indicating the reference objects.
    """
    # Connect decoding steps.
    batch_size = inputs.get_shape()[1]
    initial_state = self._core.initial_state(batch_size, trainable=False)
    _, state = tf.nn.dynamic_rnn(
        cell=self._core,
        inputs=inputs,
        sequence_length=input_sequence_length,
        time_major=True,
        initial_state=initial_state
    )
    # Connect decoding steps.
    zero_input = tf.zeros(shape=targets.get_shape())
    output_sequence, _ = tf.nn.dynamic_rnn(
        cell=self._core,
        inputs=zero_input,  # Non-autoregressive model.  Zeroed input.
        sequence_length=output_sequence_length,
        initial_state=state,
        time_major=True)
    outputs = snt.BatchApply(self._final_mlp)(output_sequence)
    logits = snt.BatchApply(snt.Linear(self._target_size))(outputs)
    tf.logging.info("Connected seq2seq model.")
    return logits


def build_and_train(iterations, log_stride, test=False):
  """Construct the data, model, loss and optimizer then train."""

  # Test mode settings.
  batch_size = 2 if test else FLAGS.batch_size
  num_mems = 2 if test else FLAGS.num_mems
  num_heads = 1 if test else FLAGS.num_mems
  num_blocks = 1 if test else FLAGS.num_mems
  head_size = 4 if test else FLAGS.head_size
  max_length = 3 if test else FLAGS.max_length
  max_nest = 2 if test else FLAGS.max_nest
  mlp_size = (20,) if test else (256, 256, 256, 256)

  with tf.Graph().as_default():
    t0 = time.time()

    # Initialize the dataset.
    lte_train = learn_to_execute.LearnToExecute(
        batch_size, max_length, max_nest)
    lte_test = learn_to_execute.LearnToExecute(
        batch_size, max_length, max_nest, mode=learn_to_execute.Mode.TEST)
    train_data_iter = lte_train.make_one_shot_iterator().get_next()
    test_data_iter = lte_test.make_one_shot_iterator().get_next()
    output_size = lte_train.state.vocab_size

    # Create the model.
    core = snt.RelationalMemory(
        mem_slots=num_mems,
        head_size=head_size,
        num_heads=num_heads,
        num_blocks=num_blocks,
        gate_style=FLAGS.gate_style)
    final_mlp = snt.nets.MLP(
        output_sizes=mlp_size,
        activate_final=True)
    model = SequenceModel(
        core=core,
        target_size=output_size,
        final_mlp=final_mlp)
    tf.logging.info("Instantiated models ({:3f})".format(time.time() - t0))

    # Define the loss & accuracy.
    def loss_fn(inputs, targets, input_sequence_length, output_sequence_length):
      """Creates the loss and the exports."""
      logits = model(
          inputs, targets, input_sequence_length, output_sequence_length)
      targets = tf.cast(targets, tf.int32)
      sq_sz_out_max = targets.shape[0].value

      # Create a mask to ignore accuracy on buffer characters.
      sequence_sizes = tf.cast(output_sequence_length, tf.float32)
      lengths_transposed = tf.expand_dims(sequence_sizes, 1)
      range_row = tf.expand_dims(
          tf.range(0, sq_sz_out_max, 1, dtype=tf.float32), 0)
      mask = tf.cast(tf.transpose(tf.less(range_row, lengths_transposed)),
                     tf.float32)

      # Compute token accuracy and solved.
      correct = tf.equal(tf.argmax(logits, 2), tf.argmax(targets, 2))
      solved = tf.reduce_all(tf.boolean_mask(correct, tf.squeeze(mask)), axis=0)
      token_acc = tf.reduce_sum(tf.cast(correct, tf.float32) * mask)
      token_acc /= tf.reduce_sum(sequence_sizes)

      # Compute Loss.
      mask = tf.cast(tf.tile(tf.expand_dims(mask, 2), (1, 1, logits.shape[2])),
                     tf.float32)
      masked_logits = logits * mask
      masked_target = tf.cast(targets, tf.float32) * mask
      logits_flat = tf.reshape(masked_logits,
                               [sq_sz_out_max * batch_size, -1])
      target_flat = tf.reshape(masked_target,
                               [sq_sz_out_max * batch_size, -1])
      xent = tf.nn.softmax_cross_entropy_with_logits(logits=logits_flat,
                                                     labels=target_flat)
      loss = tf.reduce_mean(xent)
      return loss, token_acc, solved

    # Get training step counter.
    global_step = tf.train.get_or_create_global_step()

    # Create the optimizer.
    learning_rate_op = tf.reduce_max([
        tf.train.exponential_decay(
            FLAGS.learning_rate,
            global_step,
            decay_steps=FLAGS.epochs // 100,
            decay_rate=0.9,
            staircase=False),
        FLAGS.min_learning_rate
    ])
    optimizer = tf.train.AdamOptimizer(learning_rate_op)

    # Compute loss, accuracy & the step op.
    inputs, targets, _, input_lengths, output_lengths = train_data_iter
    train_loss, train_acc, train_sol = loss_fn(
        inputs, targets, input_lengths, output_lengths)
    step_op = optimizer.minimize(train_loss, global_step=global_step)
    inputs, targets, _, input_lengths, output_lengths = test_data_iter
    _, test_acc, test_sol = loss_fn(
        inputs, targets, input_lengths, output_lengths)
    tf.logging.info("Created losses and optimizers ({:3f})".format(
        time.time() - t0))

    # Begin Training.
    t0 = time.time()
    tf.logging.info("Starting training ({:3f})".format(time.time() - t0))
    with tf.train.SingularMonitoredSession() as sess:
      for it in six.moves.range(iterations):
        sess.run([step_op, learning_rate_op])
        if it % log_stride == 0:
          loss_v, train_acc_v, test_acc_v, train_sol_v, test_sol_v = sess.run([
              train_loss, train_acc, test_acc, train_sol, test_sol])
          elapsed = time.time() - t0
          tf.logging.info(
              "iter: {:2d}, train loss {:3f}; train acc {:3f}; test acc {:3f};"
              " train solved {:3f}; test solved {:3f}; ({:3f})".format(
                  it, loss_v, train_acc_v, test_acc_v, train_sol_v, test_sol_v,
                  elapsed))


def main(unused_argv):
  build_and_train(FLAGS.epochs, FLAGS.log_stride, test=True)

if __name__ == "__main__":
  tf.app.run()
