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

"""Custom variable getter for Bayes by Backprop.

Bayes by Backprop is an algorithm for learning a probability distribution
over neural network weights. Please see :cite:`blundell2015weight` for
details. This implementation is compatible with Recurrent Neural Networks as in
:cite:`fortunato2017bayesian`.

Usage example:
  bbb_getter = snt.custom_getters.BayesByBackprop()

  # Use default posterior estimator mode (`SAMPLE`).
  with snt.custom_variable_getter(bbb_getter()):
    y = model(x)

  # Use a different posteror estimator mode.
  with snt.custom_variable_getter(bbb_getter(bbb.EstimatorMode.MEAN)):
    y = model(x)
"""
from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

import math

from absl import logging
import enum
from sonnet.src import initializers
from sonnet.src import utils
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from typing import Callable, Dict, NamedTuple, Optional, Text, Union


_OK_DTYPES_FOR_BBB = (tf.float16, tf.float32, tf.float64, tf.bfloat16)
_OK_PZATION_TYPE = tfd.FULLY_REPARAMETERIZED


class EstimatorMode(enum.Enum):
  SAMPLE = "sample"
  MEAN = "mean"


Distributions = NamedTuple(
    "Distributions",
    [("posterior", tfd.Distribution), ("prior", tfd.Distribution)])


def diagonal_gaussian_posterior_builder(
    var: tf.Variable,
    loc_initializer: Optional[initializers.Initializer] = None,
    scale_initializer: Optional[initializers.Initializer] = None) -> tfd.Normal:
  """A pre-canned builder for diagonal gaussian posterior distributions.

  Given a variable, returns a distribution object for a diagonal posterior over
  a variable of the requisite shape.

  Args:
    var: A variable.
    loc_initializer: Optional initializer for `loc` variable.
    scale_initializer: Optional initializer for `scale` variable.

  Returns:
    An instance of `tfd.Normal` representing the posterior distribution over the
    variable in question.
  """
  name = var.name[:-2]  # Strip the ":0" suffix.

  # Please see the documentation for `tfd.Normal.param_static_shapes`.
  parameter_shapes = tfd.Normal.param_static_shapes(var.shape)

  # By default, we want to initialize `loc` with the same distribution as `var`.
  # As we cannot know `var`'s initializer, we simply copy the initial value.
  if loc_initializer is None:
    assert parameter_shapes["loc"] == var.shape
    loc_initializer = lambda *_: _try_get_initial_value(var)

  if scale_initializer is None:
    scale_initializer = initializers.Constant(_inverse_softplus(0.01))

  with tf.name_scope("posterior"):
    with tf.name_scope("loc"):
      loc = tf.Variable(
          loc_initializer(parameter_shapes["loc"], var.dtype), name=name)

    with tf.name_scope("scale"):
      scale = tf.Variable(
          scale_initializer(parameter_shapes["scale"], var.dtype), name=name)

    return tfd.Normal(loc=loc, scale=tf.nn.softplus(scale), name=name)


def fixed_gaussian_prior_builder(var: tf.Variable) -> tfd.Normal:
  """A pre-canned builder for fixed gaussian prior distributions.

  Given a variable, returns a distribution object for a scalar-valued fixed
  gaussian prior which will be broadcast over a variable of the requisite shape.

  Args:
    var: A variable.

  Returns:
    An instance of `tfd.Normal` representing the prior distribution over the
    variable in question.
  """
  name = var.name[:-2]  # Strip the ":0" suffix.
  with tf.name_scope("prior"):
    return tfd.Normal(loc=0.0, scale=0.01, name=name)


def adaptive_gaussian_prior_builder(
    var: tf.Variable,
    loc_initializer: Optional[initializers.Initializer] = None,
    scale_initializer: Optional[initializers.Initializer] = None) -> tfd.Normal:
  """A pre-canned builder for adaptive scalar gaussian prior distributions.

  Given a variable, returns a distribution object for a scalar-valued adaptive
  gaussian prior which will be broadcast over a variable of the requisite shape.
  This prior's parameters (e.g `loc` and `scale` for a gaussian) will consist of
  a single learned scalar for the entire `tf.Variable` for which it serves as
  the prior, regardless of that `tf.Variable`'s shape.

  Args:
    var: A variable.
    loc_initializer: Optional initializer for `loc` variable.
    scale_initializer: Optional initializer for `scale` variable.

  Returns:
    An instance of `tfd.Normal` representing the prior distribution over the
    variable in question.
  """
  name = var.name[:-2]  # Strip the ":0" suffix.

  # By default, we want to initialize `loc` with the same distribution as `var`.
  # As we cannot know `var`'s initializer, we simply copy the initial value.
  if loc_initializer is None:
    def init_from_var(shape, dtype):
      del shape, dtype  # Unused.
      var_init_val = _try_get_initial_value(var)
      return tf.reshape(var_init_val, [-1])[0]

    loc_initializer = init_from_var

  if scale_initializer is None:
    scale_initializer = initializers.Constant(_inverse_softplus(0.01))

  with tf.name_scope("prior"):
    with tf.name_scope("loc"):
      loc = tf.Variable(loc_initializer((), var.dtype), name=name)

    with tf.name_scope("scale"):
      scale = tf.Variable(scale_initializer((), var.dtype), name=name)

    return tfd.Normal(loc=loc, scale=tf.nn.softplus(scale), name=name)


def stochastic_kl(
    posterior: tfd.Distribution,
    prior: tfd.Distribution,
    sample: tf.Tensor) -> tf.Tensor:
  """Ubiquitous stochastic KL estimator."""
  return tf.subtract(
      tf.reduce_sum(posterior.log_prob(sample)),
      tf.reduce_sum(prior.log_prob(sample)))


def analytic_kl(
    posterior: tfd.Distribution,
    prior: tfd.Distribution,
    sample: tf.Tensor) -> tf.Tensor:
  """Analytic KL divergence."""
  del sample  # Unused.
  return tf.reduce_sum(tfd.kl_divergence(posterior, prior))


DistributionBuilder = Callable[[tf.Variable], tfd.Distribution]


# TODO(b/134743802): Change to `snt.Module`. Remove explicit name scope below.
class BayesByBackprop(tf.Module):
  """A Bayes by Backprop custom variable getter builder.

  See module docs for usage example.
  """

  def __init__(
      self,
      posterior_builder: DistributionBuilder = (
          diagonal_gaussian_posterior_builder),
      prior_builder: DistributionBuilder = fixed_gaussian_prior_builder,
      name=None):
    super(BayesByBackprop, self).__init__(name=name)
    self._posterior_builder = posterior_builder
    self._prior_builder = prior_builder
    self._distributions = {}  # type: Dict[utils.CompareById[tf.Variable], Distributions]

  def __call__(self, estimator_mode: EstimatorMode = EstimatorMode.SAMPLE):
    return PosteriorEstimator(owner=self, estimator_mode=estimator_mode)

  @tf.Module.with_name_scope
  def get_or_create_distributions(self, var: tf.Variable) -> Distributions:
    """Returns distributions for the given variable (creating, as needed)."""
    if var.dtype not in _OK_DTYPES_FOR_BBB:
      raise ValueError("Disallowed data type: {}.".format(var.dtype))

    dists = self._distributions.get(utils.CompareById(var))
    if dists is None:
      posterior = self._posterior_builder(var)
      prior = self._prior_builder(var)

      if posterior.reparameterization_type != _OK_PZATION_TYPE:
        raise ValueError(
            "Distribution '{}' incompatible with Bayes by Backprop.".format(
                posterior.__class__.__name__))

      dists = Distributions(posterior, prior)
      self._distributions[utils.CompareById(var)] = dists

    return dists

  def get_distributions(self, var: tf.Variable) -> Distributions:
    """Returns the distributions for the given variable."""
    return self._distributions[utils.CompareById(var)]


# KL cost estimator function.
KLCostFn = Callable[[tfd.Distribution, tfd.Distribution, tf.Tensor], tf.Tensor]


class PosteriorEstimator(object):
  """Bayes by Backprop posterior estimator.

  This is intended to be used as a custom variable getter. See module docs for
  usage example.
  """

  def __init__(self, owner: BayesByBackprop, estimator_mode: EstimatorMode):
    self._owner = owner
    self._estimator_mode = estimator_mode
    self._estimates = {}  # type: Dict[utils.CompareById[tf.Variable], tf.Tensor]

  def __call__(self, var: tf.Variable) -> Union[tf.Variable, tf.Tensor]:
    """Returns the posterior estimate for the given variable."""
    if not var.trainable: return var

    posterior = self._owner.get_or_create_distributions(var).posterior
    estimate = _estimate(posterior, self._estimator_mode)
    self._estimates[utils.CompareById(var)] = estimate
    return estimate

  def get_total_kl_cost(
      self,
      kl_cost_fn: KLCostFn = stochastic_kl,
      name: Optional[Text] = "total_kl_cost",
      predicate: Optional[Callable[[tf.Variable], bool]] = None) -> tf.Tensor:
    """Get the total cost for all (or a subset of) the stochastic variables.

    Typically, this should be scaled by `1 / dataset_size`.

    Args:
      kl_cost_fn: The KL function.
      name: A name for the tensor representing the total KL cost.
      predicate: A callable dictating which variables are included.
        If `None`, all variables are included.

    Returns:
      A `Tensor` representing the total KL cost in the ELBO loss.
    """
    estimates = ((v.wrapped, e) for v, e in self._estimates.items())
    if predicate is not None:
      estimates = (it for it in estimates if predicate(it[0]))

    kl_costs = []
    for var, estimate in estimates:
      dists = self._owner.get_distributions(var)
      kl_costs.append(kl_cost_fn(dists.posterior, dists.prior, estimate))

    if kl_costs:
      return tf.add_n(kl_costs, name=name)
    else:
      logging.warning("No Bayes by Backprop variables found!")
      return tf.constant(0., name=name)


def _estimate(posterior: tfd.Distribution, estimator_mode: EstimatorMode):
  if estimator_mode == EstimatorMode.SAMPLE:
    return posterior.sample()
  elif estimator_mode == EstimatorMode.MEAN:
    return posterior.mean()
  else:
    raise ValueError("Unknown `estimator_mode`: {}".format(estimator_mode))


def _inverse_softplus(y):
  return math.log(math.exp(y) - 1.0)


def _try_get_initial_value(var: tf.Variable) -> tf.Tensor:
  """Returns `var`'s initial value if possible, else its current value."""
  try:
    return var.initial_value
  except RuntimeError:
    # `initial_value` is not available when running eagerly (or if we were
    # running eagerly when `var` was created).
    return var.value()
