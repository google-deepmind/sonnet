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
"""Tests for Bayes by Backprop custom getter."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import sonnet as snt
import sonnet.python.custom_getters.bayes_by_backprop as bbb
import tensorflow as tf


def softplus(x):
  return np.log(1.0 + np.exp(x))


def test_diag_gaussian_builder_builder(
    init_loc=0.0, init_scale=0.01, dist_cls=tf.distributions.Normal,
    name_append="posterior"):

  def diagonal_gaussian_posterior_builder(getter, name, *args, **kwargs):
    shape = kwargs.pop("shape")
    parameter_shapes = dist_cls.param_static_shapes(shape)
    kwargs["initializer"] = tf.constant_initializer(init_loc)
    loc_var = getter(
        name + "/{}_loc".format(name_append),
        shape=parameter_shapes["loc"],
        *args,
        **kwargs)
    kwargs["initializer"] = tf.constant_initializer(init_scale)
    scale_var = getter(
        name + "/{}_scale".format(name_append),
        shape=parameter_shapes["scale"],
        *args,
        **kwargs)
    posterior = dist_cls(
        loc=loc_var,
        scale=tf.nn.softplus(scale_var),
        name="{}_posterior_dist".format(name))
    posterior_vars = {"loc": loc_var, "scale": scale_var}
    return posterior, posterior_vars

  return diagonal_gaussian_posterior_builder


def uniform_builder(
    getter, name, *args, **kwargs):
  del kwargs["initializer"]
  shape = kwargs.pop("shape")
  parameter_shapes = tf.distributions.Uniform.param_static_shapes(shape)
  low_var = getter(
      name + "/low", shape=parameter_shapes["low"], *args, **kwargs)
  hi_var = getter(name + "/hi", shape=parameter_shapes["high"], *args, **kwargs)
  uniform_dist = tf.distributions.Uniform(low=low_var, high=hi_var)
  return uniform_dist


class BBBTest(tf.test.TestCase):

  def test_mean_mode_is_deterministic_and_correct(self):
    softplus_of_three = softplus(3.0)

    bbb_getter = bbb.bayes_by_backprop_getter(
        posterior_builder=test_diag_gaussian_builder_builder(10.9, 3.0),
        prior_builder=bbb.fixed_gaussian_prior_builder,
        sampling_mode_tensor=tf.constant(bbb.EstimatorModes.mean))

    with tf.variable_scope("my_scope", custom_getter=bbb_getter):
      my_variable = tf.get_variable("v", shape=[2], dtype=tf.float32)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    with self.test_session() as sess:
      sess.run(init_op)
      variable_value_one = sess.run(my_variable)
      variable_value_two = sess.run(my_variable)
      variable_value_three = sess.run(my_variable)
    self.assertAllClose(variable_value_one,
                        np.zeros(shape=[2]) + 10.9,
                        atol=1e-5)
    self.assertAllClose(variable_value_two,
                        np.zeros(shape=[2]) + 10.9,
                        atol=1e-5)
    self.assertAllClose(variable_value_three,
                        np.zeros(shape=[2]) + 10.9,
                        atol=1e-5)

    variable_metadata = bbb.get_variable_metadata()
    self.assertTrue(len(variable_metadata) == 1)
    q_dist_sigma = variable_metadata[0].posterior.scale

    with self.test_session() as sess:
      sigma_res = sess.run(q_dist_sigma)
    self.assertAllClose(sigma_res,
                        np.zeros(shape=[2]) + softplus_of_three,
                        atol=1e-5)

  def test_sample_mode_is_stochastic_and_can_be_switched(self):
    use_mean = tf.constant(bbb.EstimatorModes.mean)
    use_sample = tf.constant(bbb.EstimatorModes.sample)
    sampling_mode = tf.get_variable(
        "bbb_sampling_mode",
        initializer=tf.constant_initializer(bbb.EstimatorModes.sample),
        dtype=tf.string,
        shape=(),
        trainable=False)
    set_to_mean_mode = tf.assign(sampling_mode, use_mean)
    set_to_sample_mode = tf.assign(sampling_mode, use_sample)

    softplus_of_twenty = softplus(20.0)
    bbb_getter = bbb.bayes_by_backprop_getter(
        posterior_builder=test_diag_gaussian_builder_builder(10.9, 20.0),
        prior_builder=bbb.fixed_gaussian_prior_builder,
        sampling_mode_tensor=sampling_mode)

    with tf.variable_scope("my_scope", custom_getter=bbb_getter):
      my_variable = tf.get_variable("v", shape=[10, 3], dtype=tf.float32)

    # Check that the distribution has the right parameters.
    variable_metadata = bbb.get_variable_metadata()
    self.assertTrue(len(variable_metadata) == 1)
    q_dist_mean = variable_metadata[0].posterior.loc
    q_dist_sigma = variable_metadata[0].posterior.scale

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    with self.test_session() as sess:
      sess.run(init_op)
      mean_res, sigma_res = sess.run([q_dist_mean, q_dist_sigma])
      variable_value_one = sess.run(my_variable)
      variable_value_two = sess.run(my_variable)
    self.assertAllClose(mean_res, np.zeros(shape=[10, 3])+10.9)
    self.assertAllClose(sigma_res, np.zeros(shape=[10, 3]) + softplus_of_twenty)

    actual_distance = np.sqrt(
        np.sum(np.square(variable_value_one - variable_value_two)))
    expected_distance_minimum = 5
    self.assertGreater(actual_distance, expected_distance_minimum)

    # Now the value should be deterministic again.
    with self.test_session() as sess:
      sess.run(set_to_mean_mode)
      variable_value_three = sess.run(my_variable)
      variable_value_four = sess.run(my_variable)
      variable_value_five = sess.run(my_variable)
    self.assertAllClose(variable_value_three,
                        np.zeros(shape=[10, 3]) + 10.9,
                        atol=1e-5)
    self.assertAllClose(variable_value_four,
                        np.zeros(shape=[10, 3]) + 10.9,
                        atol=1e-5)
    self.assertAllClose(variable_value_five,
                        np.zeros(shape=[10, 3]) + 10.9,
                        atol=1e-5)

    # Now it should be stochastic again.
    with self.test_session() as sess:
      sess.run(set_to_sample_mode)
      variable_value_six = sess.run(my_variable)
      variable_value_seven = sess.run(my_variable)
    actual_new_distance = np.sqrt(
        np.sum(np.square(variable_value_six - variable_value_seven)))
    self.assertGreater(actual_new_distance, expected_distance_minimum)

  def test_variable_sharing(self):
    _, x_size = input_shape = [5, 5]

    sample_mode = tf.constant(bbb.EstimatorModes.sample)
    mean_mode = tf.constant(bbb.EstimatorModes.mean)
    sampling_mode = tf.get_variable(
        "bbb_sampling_mode",
        initializer=tf.constant_initializer(bbb.EstimatorModes.sample),
        dtype=tf.string,
        shape=(),
        trainable=False)
    set_to_sample_mode = tf.assign(sampling_mode, sample_mode)
    set_to_mean_mode = tf.assign(sampling_mode, mean_mode)

    bbb_getter = bbb.bayes_by_backprop_getter(
        posterior_builder=bbb.diagonal_gaussian_posterior_builder,
        prior_builder=bbb.fixed_gaussian_prior_builder,
        kl_builder=bbb.stochastic_kl_builder,
        sampling_mode_tensor=sampling_mode)

    tf.get_variable_scope().set_custom_getter(bbb_getter)
    mlp = snt.nets.MLP(output_sizes=[32, x_size])
    x_train = tf.placeholder(dtype=tf.float32, shape=input_shape)
    x_test = tf.placeholder(dtype=tf.float32, shape=input_shape)

    # Dummy targets.
    target_train = x_train + 3.0
    target_test = x_test + 3.0

    y_train = mlp(x_train)

    # Also, y_test should be deterministic for fixed x.
    y_test = mlp(x_test)

    # Expect there to be two parameter for w and b for each layer in the MLP,
    #. That's 2 * 2 * 2 = 8. But ONLY for the training set.
    expected_number_of_variables = 8
    actual_number_of_variables = len(tf.trainable_variables())
    self.assertTrue(expected_number_of_variables == actual_number_of_variables)

    loss_train = tf.reduce_sum(tf.square(y_train - target_train),
                               reduction_indices=[1])
    loss_train = tf.reduce_mean(loss_train, reduction_indices=[0])
    loss_test = tf.reduce_sum(tf.square(y_test - target_test),
                              reduction_indices=[1])
    loss_test = tf.reduce_mean(loss_test)

    kl_cost = bbb.get_total_kl_cost() * 0.000001
    total_train_loss = loss_train + kl_cost
    optimizer = tf.train.GradientDescentOptimizer(0.001)
    train_step = optimizer.minimize(total_train_loss)

    x_feed = np.random.normal(size=input_shape)
    fd = {
        x_train: x_feed,
        x_test: x_feed
    }

    init_op = tf.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(init_op)
      sess.run(set_to_mean_mode)
      y_test_res_one = sess.run(y_test, feed_dict=fd)
      y_test_res_two = sess.run(y_test, feed_dict=fd)
      sess.run(set_to_sample_mode)
    self.assertAllClose(y_test_res_one, y_test_res_two)

    n_train = 10
    check_freq = 2
    with self.test_session() as sess:
      for i in xrange(n_train):
        if i % check_freq == 0:
          sess.run(set_to_mean_mode)
          to_run = [y_train, y_test, loss_train, loss_test, kl_cost]
        else:
          to_run = [y_train, y_test, loss_train, loss_test, kl_cost, train_step]
        res = sess.run(to_run, feed_dict=fd)
        loss_train_res, loss_test_res = res[2:4]

        if i % check_freq == 0:
          self.assertAllClose(loss_train_res, loss_test_res)
          sess.run(set_to_sample_mode)

  def testLastSampleMode(self):
    """Tests that the 'last sample' estimator mode uses the last sample."""

    class CustomNormal(tf.distributions.Normal):
      """A custom normal distribution which implements `self.last_sample()`."""

      def __init__(self, *args, **kwargs):
        super(CustomNormal, self).__init__(*args, **kwargs)
        self._noise = tf.get_variable(
            name=self.loc.name.replace(":", "_") + "_noise",
            shape=self.loc.shape,
            dtype=self.loc.dtype,
            initializer=tf.random_normal_initializer(0.0, 1.0),
            trainable=False)

      def sample(self):
        noise = self._noise.assign(tf.random_normal(self.loc.shape))
        return self.last_sample(noise)

      def last_sample(self, noise=None):
        if noise is None:
          noise = self._noise
        return noise * self.scale + self.loc

    sampling_mode_tensor = tf.get_variable(
        name="sampling_mode",
        dtype=tf.string,
        shape=(),
        trainable=False,
        initializer=tf.constant_initializer(bbb.EstimatorModes.sample))
    enter_last_sample_mode = tf.assign(
        sampling_mode_tensor, tf.constant(bbb.EstimatorModes.last_sample))
    bbb_getter = bbb.bayes_by_backprop_getter(
        posterior_builder=test_diag_gaussian_builder_builder(
            dist_cls=CustomNormal),
        prior_builder=bbb.adaptive_gaussian_prior_builder,
        kl_builder=bbb.stochastic_kl_builder,
        sampling_mode_tensor=sampling_mode_tensor)
    with tf.variable_scope("model_scope", custom_getter=bbb_getter):
      model = snt.Linear(5)

    data = tf.placeholder(shape=(2, 4), dtype=tf.float32)
    outputs = model(data)

    # We expect there to be 8 trainable variables.
    # model (Linear has two variables: weight and bias).
    # The posterior has two variables (mu and sigma) for each variable.
    # So does the prior (since it's adaptive).
    self.assertEqual(len(tf.trainable_variables()), 2*2*2)

    init_op = tf.global_variables_initializer()
    x_feed = np.random.normal(size=(2, 4))
    with self.test_session() as sess:
      sess.run(init_op)
      output_res_one = sess.run(outputs, feed_dict={data: x_feed})
      output_res_two = sess.run(outputs, feed_dict={data: x_feed})
      sess.run(enter_last_sample_mode)
      output_res_three = sess.run(outputs, feed_dict={data: x_feed})
      output_res_four = sess.run(outputs, feed_dict={data: x_feed})

    # One and two should be different samples.
    self.assertTrue((output_res_one != output_res_two).all())
    # Two through four should be the same.
    self.assertAllClose(output_res_two, output_res_three)
    self.assertAllClose(output_res_three, output_res_four)
    self.assertAllClose(output_res_two, output_res_four)

  def testRecurrentNetSamplesWeightsOnce(self):
    """Test that sampling of the weights is done only once for a sequence.

    Test strategy: Provide an input sequence x whose value is the same at each
    time step. If the outputs from f_theta() are the same at each time step,
    this is evidence (but not proof) that theta is the same at each time step.
    """
    seq_length = 10
    batch_size = 1
    input_dim = 5
    output_dim = 5

    bbb_getter = bbb.bayes_by_backprop_getter(
        posterior_builder=bbb.diagonal_gaussian_posterior_builder,
        prior_builder=bbb.fixed_gaussian_prior_builder,
        kl_builder=bbb.stochastic_kl_builder,
        sampling_mode_tensor=tf.constant(bbb.EstimatorModes.sample))

    class NoStateLSTM(snt.LSTM):
      """An LSTM which ignores hidden state."""

      def _build(self, inputs, state):
        outputs, _ = super(NoStateLSTM, self)._build(inputs, state)
        return outputs, state

    with tf.variable_scope("model", custom_getter=bbb_getter):
      core = NoStateLSTM(output_dim)

    input_seq = tf.ones(shape=(seq_length, batch_size, input_dim))
    output_seq, _ = tf.nn.dynamic_rnn(
        core,
        inputs=input_seq,
        initial_state=core.initial_state(batch_size=batch_size),
        time_major=True)

    init_op = tf.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(init_op)
      output_res_one = sess.run(output_seq)
      output_res_two = sess.run(output_seq)

    # Ensure that the sequence is the same at every time step, a necessary
    # but not sufficient condition for the weights to be the same.
    output_zero = output_res_one[0]
    for time_step_output in output_res_one[1:]:
      self.assertAllClose(output_zero, time_step_output)

    # Ensure that the noise is different in the second run by checking that
    # the output sequence is different now.
    for first_run_elem, second_run_elem in zip(output_res_one, output_res_two):
      distance = np.linalg.norm(
          first_run_elem.flatten() - second_run_elem.flatten())
      self.assertGreater(distance, 0.001)

  def testFreshNoisePerConnection(self):
    """Test that the `fresh_noise_per_connection` flag works as advertised."""
    def create_custom_getter(fresh_noise_per_connection):
      bbb_getter = bbb.bayes_by_backprop_getter(
          posterior_builder=bbb.diagonal_gaussian_posterior_builder,
          prior_builder=bbb.fixed_gaussian_prior_builder,
          kl_builder=bbb.stochastic_kl_builder,
          sampling_mode_tensor=tf.constant(bbb.EstimatorModes.sample),
          fresh_noise_per_connection=fresh_noise_per_connection)
      return bbb_getter

    # 1. fresh_noise_per_connection == True.
    # test strategy: connect a module twice in sample mode and check that the
    # weights are different.
    fresh_noise_getter = create_custom_getter(fresh_noise_per_connection=True)
    with tf.variable_scope("fresh_noise", custom_getter=fresh_noise_getter):
      fresh_noise_mod = snt.Linear(3)

    x = tf.ones(shape=(3, 2))
    y_fresh_one = fresh_noise_mod(x)
    y_fresh_two = fresh_noise_mod(x)

    # 2. fresh_noise_per_connection == False.
    # test strategy: connect a module twice in sample mode and check that the
    # weights are the same.
    reuse_noise_getter = create_custom_getter(fresh_noise_per_connection=False)
    with tf.variable_scope("reuse_noise", custom_getter=reuse_noise_getter):
      reuse_noise_mod = snt.Linear(3)

    y_reuse_one = reuse_noise_mod(x)
    y_reuse_two = reuse_noise_mod(x)

    init_op = tf.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(init_op)
      yf_one, yf_two, yr_one, yr_two = sess.run([
          y_fresh_one,
          y_fresh_two,
          y_reuse_one,
          y_reuse_two
      ])

    self.assertAllClose(yr_one, yr_two)
    self.assertTrue(np.linalg.norm(yf_one - yf_two) > 0.0001)

  def testWeightsResampledWithKeepControlDeps(self):
    """Test that weights are resampled with `keep_control_dependencies=True`.

    Test strategy: We test the inverse of `testRecurrentNetSamplesWeightsOnce`.
    Provide an input sequence x whose value is the same at each time step. If
    the outputs from f_theta() are the different at each time step, then theta
    is different at each time step. In principle, it is possible that different
    thetas give the same outputs, but this is very unlikely.
    """
    seq_length = 10
    batch_size = 1
    input_dim = 5
    output_dim = 5

    bbb_getter = bbb.bayes_by_backprop_getter(
        posterior_builder=bbb.diagonal_gaussian_posterior_builder,
        prior_builder=bbb.fixed_gaussian_prior_builder,
        kl_builder=bbb.stochastic_kl_builder,
        sampling_mode_tensor=tf.constant(bbb.EstimatorModes.sample),
        keep_control_dependencies=True)

    class NoStateLSTM(snt.LSTM):
      """An LSTM which ignores hidden state."""

      def _build(self, inputs, state):
        outputs, _ = super(NoStateLSTM, self)._build(inputs, state)
        return outputs, state

    with tf.variable_scope("model", custom_getter=bbb_getter):
      core = NoStateLSTM(output_dim)

    input_seq = tf.ones(shape=(seq_length, batch_size, input_dim))
    output_seq, _ = tf.nn.dynamic_rnn(
        core,
        inputs=input_seq,
        initial_state=core.initial_state(batch_size=batch_size),
        time_major=True)

    init_op = tf.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(init_op)
      output_res_one = sess.run(output_seq)
      output_res_two = sess.run(output_seq)

    # Ensure that the sequence is different at every time step
    output_zero = output_res_one[0]
    for time_step_output in output_res_one[1:]:
      distance = np.linalg.norm(
          time_step_output.flatten() - output_zero.flatten())
      self.assertGreater(distance, 0.001)

    # Ensure that the noise is different in the second run by checking that
    # the output sequence is different now.
    for first_run_elem, second_run_elem in zip(output_res_one, output_res_two):
      distance = np.linalg.norm(
          first_run_elem.flatten() - second_run_elem.flatten())
      self.assertGreater(distance, 0.001)


if __name__ == "__main__":
  tf.test.main()
