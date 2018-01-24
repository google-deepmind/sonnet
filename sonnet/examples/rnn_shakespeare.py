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

"""Example script to train a stacked LSTM on the Tiny Shakespeare dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import sonnet as snt
from sonnet.examples import dataset_shakespeare
import tensorflow as tf


FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer("num_training_iterations", 10000,
                        "Number of iterations to train for.")
tf.flags.DEFINE_integer("report_interval", 1000,
                        "Iterations between reports (samples, valid loss).")
tf.flags.DEFINE_integer("reduce_learning_rate_interval", 2500,
                        "Iterations between learning rate reductions.")
tf.flags.DEFINE_integer("lstm_depth", 3, "Number of LSTM layers.")
tf.flags.DEFINE_integer("batch_size", 32, "Batch size for training.")
tf.flags.DEFINE_integer("num_embedding", 32, "Size of embedding layer.")
tf.flags.DEFINE_integer("num_hidden", 128, "Size of LSTM hidden layer.")
tf.flags.DEFINE_integer("truncation_length", 64, "Sequence size for training.")
tf.flags.DEFINE_integer("sample_length", 1000, "Sequence size for sampling.")
tf.flags.DEFINE_float("max_grad_norm", 5, "Gradient clipping norm limit.")
tf.flags.DEFINE_float("learning_rate", 0.1, "Optimizer learning rate.")
tf.flags.DEFINE_float("reduce_learning_rate_multiplier", 0.1,
                      "Learning rate is multiplied by this when reduced.")
tf.flags.DEFINE_float("optimizer_epsilon", 0.01,
                      "Epsilon used for Adam optimizer.")
tf.flags.DEFINE_string("checkpoint_dir", "/tmp/tf/rnn_shakespeare",
                       "Checkpointing directory.")
tf.flags.DEFINE_integer("checkpoint_interval", 500,
                        "Checkpointing step interval.")


def _configure_saver(checkpoint_dir, checkpoint_interval):
  """Returns a tf.train.CheckpointSaverHook for autosaving checkpoints."""
  saver = tf.train.Saver()
  return tf.train.CheckpointSaverHook(
      checkpoint_dir=checkpoint_dir,
      save_steps=checkpoint_interval,
      saver=saver)


def build_graph(lstm_depth=3, batch_size=32, num_embedding=32, num_hidden=128,
                truncation_length=64, sample_length=1000, max_grad_norm=5,
                initial_learning_rate=0.1, reduce_learning_rate_multiplier=0.1,
                optimizer_epsilon=0.01):
  """Constructs the computation graph."""

  # Get datasets.
  dataset_train = dataset_shakespeare.TinyShakespeareDataset(
      num_steps=truncation_length,
      batch_size=batch_size,
      subset="train",
      random=True,
      name="shake_train")

  dataset_valid = dataset_shakespeare.TinyShakespeareDataset(
      num_steps=truncation_length,
      batch_size=batch_size,
      subset="valid",
      random=False,
      name="shake_valid")

  dataset_test = dataset_shakespeare.TinyShakespeareDataset(
      num_steps=truncation_length,
      batch_size=batch_size,
      subset="test",
      random=False,
      name="shake_test")

  # Define model.
  model = TextModel(
      num_embedding=num_embedding,
      num_hidden=num_hidden,
      lstm_depth=lstm_depth,
      output_size=dataset_valid.vocab_size,
      use_dynamic_rnn=True,
      use_skip_connections=True)

  # Get the training loss.
  train_input_sequence, train_target_sequence = dataset_train()
  train_output_sequence_logits, train_final_state = model(train_input_sequence)  # pylint: disable=not-callable
  train_loss = dataset_train.cost(train_output_sequence_logits,
                                  train_target_sequence)

  # Get the validation loss.
  valid_input_sequence, valid_target_sequence = dataset_valid()
  valid_output_sequence_logits, _ = model(valid_input_sequence)  # pylint: disable=not-callable
  valid_loss = dataset_valid.cost(valid_output_sequence_logits,
                                  valid_target_sequence)

  # Get the test loss.
  test_input_sequence, test_target_sequence = dataset_test()
  test_output_sequence_logits, _ = model(test_input_sequence)  # pylint: disable=not-callable
  test_loss = dataset_test.cost(test_output_sequence_logits,
                                test_target_sequence)

  # Build graph to sample some strings during training.
  initial_logits = train_output_sequence_logits[truncation_length - 1]
  train_generated_string = model.generate_string(
      initial_logits=initial_logits,
      initial_state=train_final_state,
      sequence_length=sample_length)

  # Set up global norm clipping of gradients.
  trainable_variables = tf.trainable_variables()
  grads, _ = tf.clip_by_global_norm(
      tf.gradients(train_loss, trainable_variables), max_grad_norm)

  # Get learning rate and define annealing.
  learning_rate = tf.get_variable(
      "learning_rate",
      shape=[],
      dtype=tf.float32,
      initializer=tf.constant_initializer(initial_learning_rate),
      trainable=False)
  reduce_learning_rate = learning_rate.assign(
      learning_rate * reduce_learning_rate_multiplier)

  # Get training step counter.
  global_step = tf.get_variable(
      name="global_step",
      shape=[],
      dtype=tf.int64,
      initializer=tf.zeros_initializer(),
      trainable=False,
      collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP])

  # Define optimizer and training step.
  optimizer = tf.train.AdamOptimizer(
      learning_rate, epsilon=optimizer_epsilon)
  train_step = optimizer.apply_gradients(
      zip(grads, trainable_variables),
      global_step=global_step)

  graph_tensors = {
      "train_loss": train_loss,
      "valid_loss": valid_loss,
      "test_loss": test_loss,
      "train_generated_string": train_generated_string,
      "reduce_learning_rate": reduce_learning_rate,
      "global_step": global_step,
      "train_step": train_step
  }

  # Return dataset_train for translation to human readable text.
  return graph_tensors, dataset_train


def train(num_training_iterations, report_interval,
          reduce_learning_rate_interval):
  """Trains a deep LSTM model on the Tiny Shakespeare dataset."""

  # Build the computation graph.
  graph_tensors, dataset_train = build_graph(
      lstm_depth=FLAGS.lstm_depth, batch_size=FLAGS.batch_size,
      num_embedding=FLAGS.num_embedding, num_hidden=FLAGS.num_hidden,
      truncation_length=FLAGS.truncation_length,
      sample_length=FLAGS.sample_length, max_grad_norm=FLAGS.max_grad_norm,
      initial_learning_rate=FLAGS.learning_rate,
      reduce_learning_rate_multiplier=FLAGS.reduce_learning_rate_multiplier,
      optimizer_epsilon=FLAGS.optimizer_epsilon)

  # Configure a checkpoint saver.
  saver_hook = _configure_saver(FLAGS.checkpoint_dir,
                                FLAGS.checkpoint_interval)

  # Train the network.
  with tf.train.SingularMonitoredSession(
      hooks=[saver_hook], checkpoint_dir=FLAGS.checkpoint_dir) as sess:

    start_iteration = sess.run(graph_tensors["global_step"])

    for train_iteration in range(start_iteration, num_training_iterations):
      if (train_iteration + 1) % report_interval == 0:
        train_loss_v, valid_loss_v, _ = sess.run(
            (graph_tensors["train_loss"],
             graph_tensors["valid_loss"],
             graph_tensors["train_step"]))

        train_generated_string_v = sess.run(
            graph_tensors["train_generated_string"])

        train_generated_string_human = dataset_train.to_human_readable(
            (train_generated_string_v, 0), indices=[0])

        tf.logging.info("%d: Training loss %f. Validation loss %f. Sample = %s",
                        train_iteration,
                        train_loss_v,
                        valid_loss_v,
                        train_generated_string_human)
      else:
        train_loss_v, _ = sess.run((graph_tensors["train_loss"],
                                    graph_tensors["train_step"]))
        tf.logging.info("%d: Training loss %f.", train_iteration, train_loss_v)

      if (train_iteration + 1) % reduce_learning_rate_interval == 0:
        sess.run(graph_tensors["reduce_learning_rate"])
        tf.logging.info("Reducing learning rate.")

    test_loss = sess.run(graph_tensors["test_loss"])
    tf.logging.info("Test loss %f", test_loss)


class TextModel(snt.AbstractModule):
  """A deep LSTM model, for use on the Tiny Shakespeare dataset."""

  def __init__(self, num_embedding, num_hidden, lstm_depth, output_size,
               use_dynamic_rnn=True, use_skip_connections=True,
               name="text_model"):
    """Constructs a `TextModel`.

    Args:
      num_embedding: Size of embedding representation, used directly after the
        one-hot encoded input.
      num_hidden: Number of hidden units in each LSTM layer.
      lstm_depth: Number of LSTM layers.
      output_size: Size of the output layer on top of the DeepRNN.
      use_dynamic_rnn: Whether to use dynamic RNN unrolling. If `False`, it uses
        static unrolling. Default is `True`.
      use_skip_connections: Whether to use skip connections in the
        `snt.DeepRNN`. Default is `True`.
      name: Name of the module.
    """

    super(TextModel, self).__init__(name=name)

    self._num_embedding = num_embedding
    self._num_hidden = num_hidden
    self._lstm_depth = lstm_depth
    self._output_size = output_size
    self._use_dynamic_rnn = use_dynamic_rnn
    self._use_skip_connections = use_skip_connections

    with self._enter_variable_scope():
      self._embed_module = snt.Linear(self._num_embedding, name="linear_embed")
      self._output_module = snt.Linear(self._output_size, name="linear_output")
      self._subcores = [
          snt.LSTM(self._num_hidden, name="lstm_{}".format(i))
          for i in range(self._lstm_depth)
      ]
      if self._use_skip_connections:
        skips = []
        current_input_shape = self._num_embedding
        for lstm in self._subcores:
          input_shape = tf.TensorShape([current_input_shape])
          skip = snt.SkipConnectionCore(
              lstm,
              input_shape=input_shape,
              name="skip_{}".format(lstm.module_name))
          skips.append(skip)
          # SkipConnectionCore concatenates the input with the output, so the
          # dimensionality increases with depth.
          current_input_shape += self._num_hidden
        self._subcores = skips
      self._core = snt.DeepRNN(self._subcores, skip_connections=False,
                               name="deep_lstm")

  def _build(self, one_hot_input_sequence):
    """Builds the deep LSTM model sub-graph.

    Args:
      one_hot_input_sequence: A Tensor with the input sequence encoded as a
        one-hot representation. Its dimensions should be `[truncation_length,
        batch_size, output_size]`.

    Returns:
      Tuple of the Tensor of output logits for the batch, with dimensions
      `[truncation_length, batch_size, output_size]`, and the
      final state of the unrolled core,.
    """

    input_shape = one_hot_input_sequence.get_shape()
    batch_size = input_shape[1]

    batch_embed_module = snt.BatchApply(self._embed_module)
    input_sequence = batch_embed_module(one_hot_input_sequence)
    input_sequence = tf.nn.relu(input_sequence)

    initial_state = self._core.initial_state(batch_size)

    if self._use_dynamic_rnn:
      output_sequence, final_state = tf.nn.dynamic_rnn(
          cell=self._core,
          inputs=input_sequence,
          time_major=True,
          initial_state=initial_state)
    else:
      rnn_input_sequence = tf.unstack(input_sequence)
      output, final_state = tf.contrib.rnn.static_rnn(
          cell=self._core,
          inputs=rnn_input_sequence,
          initial_state=initial_state)
      output_sequence = tf.stack(output)

    batch_output_module = snt.BatchApply(self._output_module)
    output_sequence_logits = batch_output_module(output_sequence)

    return output_sequence_logits, final_state

  @snt.reuse_variables
  def generate_string(self, initial_logits, initial_state, sequence_length):
    """Builds sub-graph to generate a string, sampled from the model.

    Args:
      initial_logits: Starting logits to sample from.
      initial_state: Starting state for the RNN core.
      sequence_length: Number of characters to sample.

    Returns:
      A Tensor of characters, with dimensions `[sequence_length, batch_size,
      output_size]`.
    """

    current_logits = initial_logits
    current_state = initial_state

    generated_letters = []
    for _ in range(sequence_length):
      # Sample a character index from distribution.
      char_index = tf.squeeze(tf.multinomial(current_logits, 1))
      char_one_hot = tf.one_hot(char_index, self._output_size, 1.0, 0.0)
      generated_letters.append(char_one_hot)

      # Feed character back into the deep_lstm.
      gen_out_seq, current_state = self._core(
          tf.nn.relu(self._embed_module(char_one_hot)),
          current_state)
      current_logits = self._output_module(gen_out_seq)

    generated_string = tf.stack(generated_letters)

    return generated_string


def main(unused_argv):

  train(
      num_training_iterations=FLAGS.num_training_iterations,
      report_interval=FLAGS.report_interval,
      reduce_learning_rate_interval=FLAGS.reduce_learning_rate_interval)


if __name__ == "__main__":
  tf.app.run()
