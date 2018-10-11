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
"""Open Source implementation of Bayesian RNN on Penn Treebank.

Please see https://arxiv.org/pdf/1704.02798.pdf, section 7.1.

Download the Penn Treebank (PTB) dataset from:
http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

Usage: python ./brnn_ptb.py --data_path=<path_to_dataset>

Above, <path_to_dataset> is the path to the 'data' subdirectory within the
directory resulting from unpacking the .tgz file whose link is given above.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

# Dependency imports

import numpy as np
import six
import sonnet as snt
from sonnet.examples import ptb_reader
import sonnet.python.custom_getters.bayes_by_backprop as bbb
import tensorflow as tf
import tensorflow_probability as tfp

nest = tf.contrib.framework.nest
FLAGS = tf.flags.FLAGS

# Data settings.
tf.flags.DEFINE_string("data_path", "/tmp/ptb_data/data", "path to PTB data.")

# Deep LSTM settings.
tf.flags.DEFINE_integer("embedding_size", 650, "embedding size.")
tf.flags.DEFINE_integer("hidden_size", 650, "network layer size")
tf.flags.DEFINE_integer("n_layers", 2, "number of layers")

# Training settings.
tf.flags.DEFINE_integer("num_training_epochs", 70, "number of training epochs")
tf.flags.DEFINE_integer("batch_size", 20, "SGD minibatch size")
tf.flags.DEFINE_integer("unroll_steps", 35, "Truncated BPTT unroll length.")
tf.flags.DEFINE_integer("high_lr_epochs", 20, "Number of epochs with lr_start.")
tf.flags.DEFINE_float("lr_start", 1.0, "SGD learning rate initializer")
tf.flags.DEFINE_float("lr_decay", 0.9, "Polynomical decay power.")

# BBB settings.
tf.flags.DEFINE_float("prior_pi", 0.25, "Determines the prior mixture weights.")
tf.flags.DEFINE_float("prior_sigma1", np.exp(-1.0), "Prior component 1 stddev.")
tf.flags.DEFINE_float("prior_sigma2", np.exp(-7.0), "Prior component 2 stddev.")

# Logging settings.
tf.flags.DEFINE_integer("print_every_batches", 500, "Sample every x batches.")
tf.flags.DEFINE_string("logbasedir", "/tmp/bayesian_rnn", "directory for logs")
tf.flags.DEFINE_string("logsubdir", "run1", "subdirectory for this experiment.")
tf.flags.DEFINE_string(
    "mode", "train_test",
    "What mode to run in. Options: ['train_only', 'test_only', 'train_test']")


tf.logging.set_verbosity(tf.logging.INFO)


_LOADED = {}
DataOps = collections.namedtuple("DataOps", "sparse_obs sparse_target")


def _run_session_with_no_hooks(sess, *args, **kwargs):
  """Only runs of the training op should contribute to speed measurement."""
  return sess._tf_sess().run(*args, **kwargs)  # pylint: disable=protected-access


def _get_raw_data(subset):
  """Loads the data or reads it from cache."""
  raw_data = _LOADED.get(subset)
  if raw_data is not None:
    return raw_data, _LOADED["vocab"]
  else:
    train_data, valid_data, test_data, vocab = ptb_reader.ptb_raw_data(
        FLAGS.data_path)
    _LOADED.update({
        "train": np.array(train_data),
        "valid": np.array(valid_data),
        "test": np.array(test_data),
        "vocab": vocab
    })
    return _LOADED[subset], vocab


class PTB(object):
  """Wraps the PTB reader of the TensorFlow tutorial."""

  def __init__(self, subset, seq_len, batch_size, name="PTB"):
    self.raw_data, self.word2id = _get_raw_data(subset)
    self.id2word = {v: k for k, v in self.word2id.items()}
    self.seq_len = seq_len
    self.batch_size = batch_size
    self.name = name

  def to_string(self, idx_seq, join_token=" "):
    return join_token.join([self.id2word[idx] for idx in idx_seq])

  def to_string_tensor(self, time_major_idx_seq_batch):
    def p_func(input_idx_seq):
      return self.to_string(input_idx_seq)
    return tf.py_func(p_func, [time_major_idx_seq_batch[:, 0]], tf.string)

  def __call__(self):
    x_bm, y_bm = ptb_reader.ptb_producer(
        self.raw_data, self.batch_size, self.seq_len, name=self.name)
    x_tm = tf.transpose(x_bm, [1, 0])
    y_tm = tf.transpose(y_bm, [1, 0])
    return DataOps(sparse_obs=x_tm, sparse_target=y_tm)

  @property
  def num_batches(self):
    return np.prod(self.raw_data.shape) // (self.seq_len * self.batch_size)

  @property
  def vocab_size(self):
    return len(self.word2id)


class GlobalNormClippingOptimizer(tf.train.Optimizer):
  """Optimizer that clips gradients by global norm."""

  def __init__(self,
               opt,
               clip_norm,
               use_locking=False,
               name="GlobalNormClippingOptimizer"):
    super(GlobalNormClippingOptimizer, self).__init__(use_locking, name)

    self._opt = opt
    self._clip_norm = clip_norm

  def compute_gradients(self, *args, **kwargs):
    return self._opt.compute_gradients(*args, **kwargs)

  def apply_gradients(self, grads_and_vars, *args, **kwargs):
    if self._clip_norm == np.inf:
      return self._opt.apply_gradients(grads_and_vars, *args, **kwargs)
    grads, vars_ = zip(*grads_and_vars)
    clipped_grads, _ = tf.clip_by_global_norm(grads, self._clip_norm)
    return self._opt.apply_gradients(zip(clipped_grads, vars_), *args, **kwargs)


class CustomScaleMixture(object):
  """A convenience class for the scale mixture."""

  def __init__(self, pi, sigma1, sigma2):
    self.mu, self.pi, self.sigma1, self.sigma2 = map(
        np.float32, (0.0, pi, sigma1, sigma2))

  def log_prob(self, x):
    n1 = tfp.distributions.Normal(self.mu, self.sigma1)
    n2 = tfp.distributions.Normal(self.mu, self.sigma2)
    mix1 = tf.reduce_sum(n1.log_prob(x), -1) + tf.log(self.pi)
    mix2 = tf.reduce_sum(n2.log_prob(x), -1) + tf.log(np.float32(1.0 - self.pi))
    prior_mix = tf.stack([mix1, mix2])
    lse_mix = tf.reduce_logsumexp(prior_mix, [0])
    return tf.reduce_sum(lse_mix)


def custom_scale_mixture_prior_builder(getter, name, *args, **kwargs):
  """A builder for the gaussian scale-mixture prior of Fortunato et al.

  Please see https://arxiv.org/abs/1704.02798, section 7.1

  Args:
    getter: The `getter` passed to a `custom_getter`. Please see the
      documentation for `tf.get_variable`.
    name: The `name` argument passed to `tf.get_variable`.
    *args: Positional arguments forwarded by `tf.get_variable`.
    **kwargs: Keyword arguments forwarded by `tf.get_variable`.

  Returns:
    An instance of `tfp.distributions.Distribution` representing the
    prior distribution over the variable in question.
  """
  # This specific prior formulation doesn't need any of the arguments forwarded
  # from `get_variable`.
  del getter
  del name
  del args
  del kwargs
  return CustomScaleMixture(
      FLAGS.prior_pi, FLAGS.prior_sigma1, FLAGS.prior_sigma2)


def lstm_posterior_builder(getter, name, *args, **kwargs):
  """A builder for a particular diagonal gaussian posterior.

  Args:
    getter: The `getter` passed to a `custom_getter`. Please see the
      documentation for `tf.get_variable`.
    name: The `name` argument passed to `tf.get_variable`.
    *args: Positional arguments forwarded by `tf.get_variable`.
    **kwargs: Keyword arguments forwarded by `tf.get_variable`.

  Returns:
    An instance of `tfp.distributions.Distribution` representing the
    posterior distribution over the variable in question.
  """
  del args
  parameter_shapes = tfp.distributions.Normal.param_static_shapes(
      kwargs["shape"])

  # The standard deviation of the scale mixture prior.
  prior_stddev = np.sqrt(
      FLAGS.prior_pi * np.square(FLAGS.prior_sigma1) +
      (1 - FLAGS.prior_pi) * np.square(FLAGS.prior_sigma2))

  loc_var = getter(
      "{}/posterior_loc".format(name),
      shape=parameter_shapes["loc"],
      initializer=kwargs.get("initializer"),
      dtype=tf.float32)
  scale_var = getter(
      "{}/posterior_scale".format(name),
      initializer=tf.random_uniform(
          minval=np.log(np.exp(prior_stddev / 4.0) - 1.0),
          maxval=np.log(np.exp(prior_stddev / 2.0) - 1.0),
          dtype=tf.float32,
          shape=parameter_shapes["scale"]))
  return tfp.distributions.Normal(
      loc=loc_var,
      scale=tf.nn.softplus(scale_var) + 1e-5,
      name="{}/posterior_dist".format(name))


def non_lstm_posterior_builder(getter, name, *args, **kwargs):
  """A builder for a particular diagonal gaussian posterior.

  Args:
    getter: The `getter` passed to a `custom_getter`. Please see the
      documentation for `tf.get_variable`.
    name: The `name` argument passed to `tf.get_variable`.
    *args: Positional arguments forwarded by `tf.get_variable`.
    **kwargs: Keyword arguments forwarded by `tf.get_variable`.

  Returns:
    An instance of `tfp.distributions.Distribution` representing the
    posterior distribution over the variable in question.
  """
  del args
  parameter_shapes = tfp.distributions.Normal.param_static_shapes(
      kwargs["shape"])

  # The standard deviation of the scale mixture prior.
  prior_stddev = np.sqrt(
      FLAGS.prior_pi * np.square(FLAGS.prior_sigma1) +
      (1 - FLAGS.prior_pi) * np.square(FLAGS.prior_sigma2))

  loc_var = getter(
      "{}/posterior_loc".format(name),
      shape=parameter_shapes["loc"],
      initializer=kwargs.get("initializer"),
      dtype=tf.float32)
  scale_var = getter(
      "{}/posterior_scale".format(name),
      initializer=tf.random_uniform(
          minval=np.log(np.exp(prior_stddev / 2.0) - 1.0),
          maxval=np.log(np.exp(prior_stddev / 1.0) - 1.0),
          dtype=tf.float32,
          shape=parameter_shapes["scale"]))
  return tfp.distributions.Normal(
      loc=loc_var,
      scale=tf.nn.softplus(scale_var) + 1e-5,
      name="{}/posterior_dist".format(name))


def build_modules(is_training, vocab_size):
  """Construct the modules used in the graph."""

  # Construct the custom getter which implements Bayes by Backprop.
  if is_training:
    estimator_mode = tf.constant(bbb.EstimatorModes.sample)
  else:
    estimator_mode = tf.constant(bbb.EstimatorModes.mean)
  lstm_bbb_custom_getter = bbb.bayes_by_backprop_getter(
      posterior_builder=lstm_posterior_builder,
      prior_builder=custom_scale_mixture_prior_builder,
      kl_builder=bbb.stochastic_kl_builder,
      sampling_mode_tensor=estimator_mode)
  non_lstm_bbb_custom_getter = bbb.bayes_by_backprop_getter(
      posterior_builder=non_lstm_posterior_builder,
      prior_builder=custom_scale_mixture_prior_builder,
      kl_builder=bbb.stochastic_kl_builder,
      sampling_mode_tensor=estimator_mode)

  embed_layer = snt.Embed(
      vocab_size=vocab_size,
      embed_dim=FLAGS.embedding_size,
      custom_getter=non_lstm_bbb_custom_getter,
      name="input_embedding")

  cores = [snt.LSTM(FLAGS.hidden_size,
                    custom_getter=lstm_bbb_custom_getter,
                    forget_bias=0.0,
                    name="lstm_layer_{}".format(i))
           for i in six.moves.range(FLAGS.n_layers)]
  rnn_core = snt.DeepRNN(
      cores,
      skip_connections=False,
      name="deep_lstm_core")

  # Do BBB on weights but not biases of output layer.
  output_linear = snt.Linear(
      vocab_size, custom_getter={"w": non_lstm_bbb_custom_getter})
  return embed_layer, rnn_core, output_linear


def build_logits(data_ops, embed_layer, rnn_core, output_linear, name_prefix):
  """This is the core model logic.

  Unrolls a Bayesian RNN over the given sequence.

  Args:
    data_ops: A `sequence_data.SequenceDataOps` namedtuple.
    embed_layer: A `snt.Embed` instance.
    rnn_core: A `snt.RNNCore` instance.
    output_linear: A `snt.Linear` instance.
    name_prefix: A string to use to prefix local variable names.

  Returns:
    A 3D time-major tensor representing the model's logits for a sequence of
    predictions. Shape `[time_steps, batch_size, vocab_size]`.
  """
  # Embed the input index sequence.
  embedded_input_seq = snt.BatchApply(
      embed_layer, name="input_embed_seq")(data_ops.sparse_obs)

  # Construct variables for holding the RNN state.
  initial_rnn_state = nest.map_structure(
      lambda t: tf.get_local_variable(  # pylint: disable long lambda warning
          "{}/rnn_state/{}".format(name_prefix, t.op.name), initializer=t),
      rnn_core.initial_state(FLAGS.batch_size))
  assign_zero_rnn_state = nest.map_structure(
      lambda x: x.assign(tf.zeros_like(x)), initial_rnn_state)
  assign_zero_rnn_state = tf.group(*nest.flatten(assign_zero_rnn_state))

  # Unroll the RNN core over the sequence.
  rnn_output_seq, rnn_final_state = tf.nn.dynamic_rnn(
      cell=rnn_core,
      inputs=embedded_input_seq,
      initial_state=initial_rnn_state,
      time_major=True)

  # Persist the RNN state for the next unroll.
  update_rnn_state = nest.map_structure(
      tf.assign, initial_rnn_state, rnn_final_state)
  with tf.control_dependencies(nest.flatten(update_rnn_state)):
    rnn_output_seq = tf.identity(rnn_output_seq, name="rnn_output_seq")
  output_logits = snt.BatchApply(
      output_linear, name="output_embed_seq")(rnn_output_seq)
  return output_logits, assign_zero_rnn_state


def build_loss(model_logits, sparse_targets):
  """Compute the log loss given predictions and targets."""
  time_major_shape = [FLAGS.unroll_steps, FLAGS.batch_size]
  flat_batch_shape = [FLAGS.unroll_steps * FLAGS.batch_size, -1]
  xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=tf.reshape(model_logits, flat_batch_shape),
      labels=tf.reshape(sparse_targets, flat_batch_shape[:-1]))
  xent = tf.reshape(xent, time_major_shape)
  # Sum over the sequence.
  sequence_neg_log_prob = tf.reduce_sum(xent, axis=0)
  # Average over the batch.
  return tf.reduce_mean(sequence_neg_log_prob, axis=0)


def train(logdir):
  """Run a network on the PTB training set, checkpointing the weights."""

  ptb_train = PTB(
      name="ptb_train",
      subset="train",
      seq_len=FLAGS.unroll_steps,
      batch_size=FLAGS.batch_size)

  # Connect to training set.
  data_ops = ptb_train()
  embed_layer, rnn_core, output_linear = build_modules(
      is_training=True, vocab_size=ptb_train.vocab_size)
  prediction_logits, zero_state_op = build_logits(
      data_ops, embed_layer, rnn_core, output_linear, name_prefix="train")
  data_loss = build_loss(prediction_logits, data_ops.sparse_target)

  # Add the KL cost.
  total_kl_cost = bbb.get_total_kl_cost()
  num_dataset_elements = FLAGS.batch_size * ptb_train.num_batches
  scaled_kl_cost = total_kl_cost / num_dataset_elements
  total_loss = tf.add(scaled_kl_cost, data_loss)

  # Optimize as usual.
  global_step = tf.get_variable(
      "num_weight_updates",
      initializer=tf.constant(0, dtype=tf.int32, shape=()),
      collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP])

  learning_rate = tf.get_variable(
      "lr", initializer=tf.constant(FLAGS.lr_start, shape=(), dtype=tf.float32))
  learning_rate_update = learning_rate.assign(learning_rate * FLAGS.lr_decay)
  optimizer = tf.train.GradientDescentOptimizer(
      learning_rate=learning_rate)
  optimizer = GlobalNormClippingOptimizer(optimizer, clip_norm=5.0)

  with tf.control_dependencies([optimizer.minimize(total_loss)]):
    global_step_and_train = global_step.assign_add(1)

  # Connect to valid set.
  ptb_valid = PTB(
      name="ptb_valid",
      subset="valid",
      seq_len=FLAGS.unroll_steps,
      batch_size=FLAGS.batch_size)
  valid_data_ops = ptb_valid()
  valid_logits, zero_valid_state = build_logits(
      valid_data_ops, embed_layer, rnn_core, output_linear, name_prefix="valid")
  valid_loss = build_loss(valid_logits, valid_data_ops.sparse_target)

  # Compute metrics for the sake of monitoring training.
  predictions = tf.cast(
      tf.argmax(prediction_logits, axis=-1), tf.int32, name="pred")
  correct_prediction_mask = tf.cast(
      tf.equal(predictions, data_ops.sparse_target), tf.int32)
  accuracy = tf.reduce_mean(
      tf.cast(correct_prediction_mask, tf.float32), name="acc")
  error_rate = tf.subtract(1.0, accuracy, name="err")
  label_probs = tf.nn.softmax(prediction_logits, dim=-1)
  predictive_entropy = tf.reduce_mean(
      label_probs * tf.log(label_probs + 1e-12) * -1.0)

  # Create tf.summary ops.
  log_ops_to_run = {
      "scalar": collections.OrderedDict([
          ("task_loss", data_loss),
          ("train_err_rate", error_rate),
          ("pred_entropy", predictive_entropy),
          ("learning_rate", learning_rate),
          ("elbo_loss", total_loss),
          ("kl_cost", total_kl_cost),
          ("scaled_kl_cost", scaled_kl_cost),
      ]),
      "text": collections.OrderedDict([
          ("labels", ptb_train.to_string_tensor(data_ops.sparse_target)),
          ("predictions", ptb_train.to_string_tensor(predictions))
      ])
  }

  for name, tensor in log_ops_to_run["scalar"].items():
    tf.summary.scalar(os.path.join("train", name), tensor)

  # The remaining logic runs the training loop and logging.
  summary_writer = tf.summary.FileWriterCache.get(logdir=logdir)
  tf.logging.info(
      "Beginning training for {} epochs, each with {} batches.".format(
          FLAGS.num_training_epochs, ptb_train.num_batches))
  with tf.train.MonitoredTrainingSession(
      is_chief=True, checkpoint_dir=logdir, save_summaries_secs=10) as sess:
    num_updates_v = _run_session_with_no_hooks(sess, global_step)
    epoch_idx_start, step_idx_start = divmod(
        num_updates_v, ptb_train.num_batches)
    tf.logging.info("On start, epoch: {}\t step: {}".format(
        epoch_idx_start, step_idx_start))
    for epoch_idx in six.moves.range(epoch_idx_start,
                                     FLAGS.num_training_epochs):
      tf.logging.info("Beginning Epoch {}/{}".format(
          epoch_idx, FLAGS.num_training_epochs))
      tf.logging.info(
          ("Beginning by evaluating on the validation set, which has "
           "{} batches.".format(ptb_valid.num_batches)))
      valid_cost = 0
      valid_steps = 0
      _run_session_with_no_hooks(sess, zero_valid_state)
      for _ in six.moves.range(ptb_valid.num_batches):
        valid_cost_v, num_updates_v = _run_session_with_no_hooks(
            sess, [valid_loss, global_step])
        valid_cost += valid_cost_v
        valid_steps += FLAGS.unroll_steps
      tf.logging.info("Validation set perplexity: {}".format(
          np.exp(valid_cost / valid_steps)))
      summary = tf.summary.Summary()
      summary.value.add(
          tag="valid/word_level_perplexity",
          simple_value=np.exp(valid_cost / valid_steps))
      summary_writer.add_summary(summary, num_updates_v)

      # Run a training epoch.
      epoch_cost = 0
      epoch_steps = 0
      for batch_idx in six.moves.range(step_idx_start, ptb_train.num_batches):
        scalars_res, num_updates_v = sess.run(
            [log_ops_to_run["scalar"], global_step_and_train])
        epoch_cost += scalars_res["task_loss"]
        epoch_steps += FLAGS.unroll_steps
        if (batch_idx - 1) % FLAGS.print_every_batches == 0:
          summary = tf.summary.Summary()
          summary.value.add(
              tag="train/word_level_perplexity",
              simple_value=np.exp(epoch_cost / epoch_steps))
          summary_writer.add_summary(summary, num_updates_v)
          scalars_res, strings_res = _run_session_with_no_hooks(
              sess, [log_ops_to_run["scalar"], log_ops_to_run["text"]])
          tf.logging.info("Num weight updates: {}".format(num_updates_v))
          for name, result in six.iteritems(scalars_res):
            tf.logging.info("{}: {}".format(name, result))
          for name, result in six.iteritems(strings_res):
            tf.logging.info("{}: {}".format(name, result))

      word_level_perplexity = np.exp(epoch_cost / epoch_steps)
      tf.logging.info(
          "Train Perplexity after Epoch {}: {}".format(
              epoch_idx, word_level_perplexity))

      end_of_epoch_fetches = [zero_state_op]
      if epoch_idx >= FLAGS.high_lr_epochs:
        end_of_epoch_fetches.append(learning_rate_update)
      _run_session_with_no_hooks(sess, end_of_epoch_fetches)

  tf.logging.info("Done training. Thanks for your time.")


def test(logdir):
  """Run a network on the PTB test set, restoring from the latest checkpoint."""
  global_step = tf.get_variable(
      "num_weight_updates",
      initializer=tf.constant(0, dtype=tf.int32, shape=()),
      collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP])

  ptb_test = PTB(
      name="ptb_test",
      subset="test",
      seq_len=FLAGS.unroll_steps,
      batch_size=FLAGS.batch_size)

  # Connect to test set.
  data_ops = ptb_test()
  # The variables in these modules will be restored from the checkpoint.
  embed_layer, rnn_core, output_linear = build_modules(
      is_training=False, vocab_size=ptb_test.vocab_size)
  prediction_logits, _ = build_logits(
      data_ops, embed_layer, rnn_core, output_linear, name_prefix="test")
  avg_nats_per_sequence = build_loss(prediction_logits, data_ops.sparse_target)

  dataset_cost = 0
  dataset_iters = 0
  with tf.train.SingularMonitoredSession(checkpoint_dir=logdir) as sess:
    tf.logging.info("Running on test set in {} batches.".format(
        ptb_test.num_batches))
    tf.logging.info("The model has trained for {} steps.".format(
        _run_session_with_no_hooks(sess, global_step)))
    for _ in range(ptb_test.num_batches):
      dataset_cost += _run_session_with_no_hooks(sess, avg_nats_per_sequence)
      dataset_iters += FLAGS.unroll_steps
    tf.logging.info("Final test set perplexity: {}.".format(
        np.exp(dataset_cost / dataset_iters)))


def main(unused_argv):
  logdir = os.path.join(FLAGS.logbasedir, FLAGS.logsubdir)
  tf.logging.info("Log Directory: {}".format(logdir))
  if FLAGS.mode == "train_only":
    train(logdir)
  elif FLAGS.mode == "test_only":
    test(logdir)
  elif FLAGS.mode == "train_test":
    tf.logging.info("Beginning a training phase of {} epochs.".format(
        FLAGS.num_training_epochs))
    train(logdir)
    tf.logging.info("Beginning testing phase.")
    with tf.Graph().as_default():
      # Enter new default graph so that we can read variables from checkpoint
      # without getting hit by name uniquification of sonnet variables.
      test(logdir)
  else:
    raise ValueError("Invalid mode {}. Please choose one of {}.".format(
        FLAGS.mode, "['train_only', 'test_only', 'train_test']"))

if __name__ == "__main__":
  tf.app.run()
