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

This is a reduced size version of the "Nth Farthest" task defined in:

  https://arxiv.org/abs/1806.01822

This resource intensive task and is advisable to run on GPU with 16GB RAM.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
# Dependency imports

from absl import flags
import six

import sonnet as snt
from sonnet.examples import dataset_nth_farthest
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
flags.DEFINE_integer("num_objects", 4, "Number of objects per dataset sample.")
flags.DEFINE_integer("num_features", 4, "Feature size per object.")
flags.DEFINE_integer("epochs", 1000000, "Total training epochs.")
flags.DEFINE_integer("log_stride", 100, "Iterations between reports.")


class SequenceModel(snt.AbstractModule):
  """Model to process n-th farthest sequence batches."""

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

  def _build(self, inputs):
    """Dynamic unroll across input objects.

    Args:
      inputs: tensor (batch x num_objects x feature). Objects to sort.

    Returns:
      Tensor (batch x num_objects); logits indicating the reference objects.
    """
    batch_size = inputs.get_shape()[0]
    output_sequence, _ = tf.nn.dynamic_rnn(
        cell=self._core,
        inputs=inputs,
        time_major=False,
        initial_state=self._core.initial_state(
            batch_size, trainable=False)
    )
    outputs = snt.BatchFlatten()(output_sequence[:, -1, :])
    outputs = self._final_mlp(outputs)
    logits = snt.Linear(self._target_size)(outputs)
    return logits


def build_and_train(iterations, log_stride, test=False):
  """Construct the data, model, loss and optimizer then train."""

  # Test mode settings.
  batch_size = 2 if test else FLAGS.batch_size
  num_mems = 2 if test else FLAGS.num_mems
  num_heads = 1 if test else FLAGS.num_mems
  num_blocks = 1 if test else FLAGS.num_mems
  head_size = 4 if test else FLAGS.head_size
  num_objects = 2 if test else FLAGS.num_objects
  num_features = 4 if test else FLAGS.num_features
  mlp_size = (20,) if test else (256, 256, 256, 256)

  with tf.Graph().as_default():
    t0 = time.time()

    # Initialize the dataset.
    dataset = dataset_nth_farthest.NthFarthest(
        batch_size, num_objects, num_features)

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
        target_size=num_objects,
        final_mlp=final_mlp)

    tf.logging.info("Instantiated models ({:3f})".format(time.time() - t0))

    # Get train and test data.
    inputs_train, labels_train = dataset.get_batch()
    inputs_test, labels_test = dataset.get_batch()

    # Define target accuracy.
    def compute_accuracy(logits, targets, name="accuracy"):
      correct_pred = tf.cast(
          tf.equal(tf.cast(targets, tf.int64), tf.argmax(logits, 1)),
          tf.float32)
      return tf.reduce_mean(correct_pred, name=name)

    # Define the loss & accuracy.
    def loss_fn(inputs, labels):
      """Creates the loss and the exports."""
      logits = model(inputs)
      labels = tf.cast(labels, tf.int32)
      loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits, labels=labels))
      accuracy = compute_accuracy(logits, labels)
      return loss, accuracy

    # Get training step counter.
    global_step = tf.get_variable(
        name="global_step",
        shape=[],
        dtype=tf.int64,
        initializer=tf.zeros_initializer(),
        trainable=False,
        collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP])

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
    train_loss, _ = loss_fn(inputs_train, labels_train)
    step_op = optimizer.minimize(train_loss, global_step=global_step)

    # Compute test accuracy
    logits_test = model(inputs_test)
    labels_test = tf.cast(labels_test, tf.int32)
    test_acc = compute_accuracy(logits_test, labels_test)

    tf.logging.info("Created losses and optimizers ({:3f})".format(
        time.time() - t0))

    # Begin Training.
    t0 = time.time()
    train_losses = []
    steps = []
    test_accs = []
    tf.logging.info("Starting training ({:3f})".format(time.time() - t0))
    with tf.train.SingularMonitoredSession() as sess:
      for it in six.moves.range(iterations):
        sess.run([step_op, learning_rate_op])
        if it % log_stride == 0:
          loss_v, acc_v = sess.run([train_loss, test_acc])
          elapsed = time.time() - t0
          tf.logging.info(
              "iter: {:2d}, train loss {:3f}; test acc {:3f} ({:3f})".format(
                  it, loss_v, acc_v, elapsed))
          train_losses.append(loss_v)
          steps.append(it)
        test_accs.append(acc_v)
  return steps, train_losses, test_accs


def main(unused_argv):
  build_and_train(FLAGS.epochs, FLAGS.log_stride)

if __name__ == "__main__":
  tf.app.run()
