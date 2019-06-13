# Copyright 2019 The Sonnet Authors. All Rights Reserved.
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
from sonnet.src import base
from sonnet.src import bayes_by_backprop as bbb
from sonnet.src import custom_getter
from sonnet.src import linear
from sonnet.src import sgd
from sonnet.src import test_utils
import tensorflow as tf
import tensorflow_probability as tfp


def _softplus(x):
  return np.log(1.0 + np.exp(x))


def _test_diag_gaussian_builder_builder(
    initial_loc, initial_scale, dist_class=tfp.distributions.Normal):
  def diagonal_gaussian_posterior_builder(var):
    name = var.name[:-2]  # Strip the ":0" suffix.
    parameter_shapes = dist_class.param_static_shapes(var.shape)

    loc = tf.Variable(
        initial_value=tf.constant(
            initial_loc, shape=parameter_shapes["loc"], dtype=var.dtype))

    scale = tf.Variable(
        initial_value=tf.constant(
            initial_scale, shape=parameter_shapes["scale"], dtype=var.dtype))

    return dist_class(
        loc=loc,
        scale=tf.nn.softplus(scale),
        name="{}/posterior_dist".format(name))

  return diagonal_gaussian_posterior_builder


class BayesByBackpropTest(test_utils.TestCase):

  def test_mean_mode_is_deterministic_and_correct(self):
    softplus_of_three = _softplus(3.0)

    bbb_getter = bbb.BayesByBackprop(
        posterior_builder=_test_diag_gaussian_builder_builder(10.9, 3.0),
        prior_builder=bbb.fixed_gaussian_prior_builder)

    class MyVariables(base.Module):
      v = tf.Variable(tf.random.normal([2]), dtype=tf.float32)

    my_variables = MyVariables()

    with custom_getter.custom_variable_getter(
        bbb_getter(bbb.EstimatorMode.MEAN)):
      expected = [10.9, 10.9]
      self.assertAllClose(expected, self.evaluate(my_variables.v), atol=1e-5)
      self.assertAllClose(expected, self.evaluate(my_variables.v), atol=1e-5)
      self.assertAllClose(expected, self.evaluate(my_variables.v), atol=1e-5)

    self.assertLen(bbb_getter._distributions, 1)
    posterior = bbb_getter.get_distributions(my_variables.v).posterior

    self.assertAllClose(
        [softplus_of_three, softplus_of_three],
        self.evaluate(posterior.scale),
        atol=1e-5)

  def test_sample_mode_is_stochastic_and_can_be_switched(self):
    softplus_of_twenty = _softplus(20.0)

    bbb_getter = bbb.BayesByBackprop(
        posterior_builder=_test_diag_gaussian_builder_builder(10.9, 20.0),
        prior_builder=bbb.fixed_gaussian_prior_builder)

    class MyVariables(base.Module):
      v = tf.Variable(tf.random.normal([10, 3]), dtype=tf.float32)

    my_variables = MyVariables()

    with custom_getter.custom_variable_getter(bbb_getter()):
      variable_value_one = self.evaluate(my_variables.v)
      variable_value_two = self.evaluate(my_variables.v)

    actual_distance = np.sqrt(
        np.sum(np.square(variable_value_one - variable_value_two)))
    self.assertGreater(actual_distance, 5)

    # Check that the distribution has the right parameters.
    self.assertLen(bbb_getter._distributions, 1)
    posterior = bbb_getter.get_distributions(my_variables.v).posterior

    expected = np.full([10, 3], 10.9)
    self.assertAllClose(expected, self.evaluate(posterior.loc))
    self.assertAllClose(
        np.full([10, 3], softplus_of_twenty), self.evaluate(posterior.scale))

    # Now the value should be deterministic again.
    with custom_getter.custom_variable_getter(
        bbb_getter(bbb.EstimatorMode.MEAN)):
      self.assertAllClose(expected, self.evaluate(my_variables.v), atol=1e-5)
      self.assertAllClose(expected, self.evaluate(my_variables.v), atol=1e-5)
      self.assertAllClose(expected, self.evaluate(my_variables.v), atol=1e-5)

    # Now it should be stochastic again.
    with custom_getter.custom_variable_getter(bbb_getter()):
      variable_value_three = self.evaluate(my_variables.v)
      variable_value_four = self.evaluate(my_variables.v)

    actual_distance = np.sqrt(
        np.sum(np.square(variable_value_three - variable_value_four)))
    self.assertGreater(actual_distance, 5)

  def test_variable_sharing(self):
    input_shape = [5, 5]

    bbb_getter = bbb.BayesByBackprop(
        posterior_builder=bbb.diagonal_gaussian_posterior_builder,
        prior_builder=bbb.fixed_gaussian_prior_builder,
        name="bbb")

    lin1 = linear.Linear(32, name="lin1")
    lin2 = linear.Linear(input_shape[1], name="lin2")
    mlp = lambda inputs: lin2(lin1(inputs))

    created_vars = []
    def record_created_vars(next_creator, **kwargs):
      var = next_creator(**kwargs)
      created_vars.append(var.name)
      return var

    with tf.variable_creator_scope(record_created_vars):
      with custom_getter.custom_variable_getter(bbb_getter()):
        mlp(tf.constant(42., shape=input_shape))
        mlp(tf.constant(43., shape=input_shape))

    expected_vars = [
        "lin1/w:0",
        "lin1/b:0",
        "lin2/w:0",
        "lin2/b:0",
        "bbb/posterior/loc/lin1/w:0",
        "bbb/posterior/scale/lin1/w:0",
        "bbb/posterior/loc/lin1/b:0",
        "bbb/posterior/scale/lin1/b:0",
        "bbb/posterior/loc/lin2/w:0",
        "bbb/posterior/scale/lin2/w:0",
        "bbb/posterior/loc/lin2/b:0",
        "bbb/posterior/scale/lin2/b:0"]

    self.assertCountEqual(expected_vars, created_vars)

    def loss_fn(x, estimator_mode=bbb.EstimatorMode.MEAN, is_training=False):
      posterior_estimator = bbb_getter(estimator_mode)

      with custom_getter.custom_variable_getter(posterior_estimator):
        y = mlp(x)

      target = x + 3.0  # Dummy target.
      loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - target), axis=1))

      if is_training:
        loss += posterior_estimator.get_total_kl_cost() * 0.000001

      return loss

    x = tf.random.normal(shape=input_shape)
    y_test = lambda: loss_fn(x)

    optimizer = sgd.SGD(0.001)

    def y_train():
      with tf.GradientTape() as tape:
        loss = loss_fn(x, bbb.EstimatorMode.SAMPLE, is_training=True)

      params = tape.watched_variables()
      grads = tape.gradient(loss, params)
      optimizer.apply(grads, params)
      return loss

    for _ in range(5):
      y_test_res_one = self.evaluate(y_test)
      y_test_res_two = self.evaluate(y_test)
      self.assertAllClose(y_test_res_one, y_test_res_two)

      self.evaluate(y_train)


if __name__ == "__main__":
  # tf.enable_v2_behavior()
  tf.test.main()
