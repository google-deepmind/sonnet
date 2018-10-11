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
"""Custom getters for Sonnet-compatible bayes by backprop.

## Algorithm Description

Bayes by Backprop is an algorithm for learning a probability distribution
over neural network weights. Please see https://arxiv.org/abs/1505.05424 for
details. This implementation is compatible with Recurrent Neural Networks as in
https://arxiv.org/abs/1704.02798.

## Usage

A minimal example is demonstrated below. A full example can be found in the
Bayesian RNN script here: https://github.com/deepmind/sonnet/tree/master/sonnet/examples/brnn_ptb.py.
```
import sonnet as snt
import sonnet.python.custom_getters.bayes_by_backprop as bbb
import tensorflow as tf

# Use a custom prior builder.
def custom_prior_builder(getter, name, *args, **kwargs):
  return tfp.distributions.Normal(0.0, 0.01)

# Use pre-canned builders for diagonal gaussian posterior and stochastic KL.
get_bbb_variable_fn = bbb.bayes_by_backprop_getter(
    prior_builder=custom_prior_builder,
    posterior_builder=bbb.diagonal_gaussian_posterior_builder,
    kl_builder=bbb.stochastic_kl_builder)

# Demonstration of how to use custom_getters with variable scopes.
with tf.variable_scope('network', custom_getter=get_bbb_variable_fn):
  model = snt.Linear(4)
  # This approach is compatible with all `tf.Variable`s constructed with
  # `tf.get_variable()`, not just those contained in sonnet modules.
  noisy_variable = tf.get_variable('w', shape=(5,), dtype=tf.float32)

# An alternative way to use BBB with sonnet modules is to use their custom
# getter argument.
model2 = snt.Linear(5, custom_getter=get_bbb_variable_fn)

# Proceed with the rest of the graph as usual.
input_data, target_data = tf.random_normal((3, 2)), tf.random_normal((3, 4))
loss = tf.reduce_sum(tf.square(model(input_data) - target_data))

# Add the scaled KL cost to the loss.
# A good choice of scaling is to divide by the number of training examples.
# See https://arxiv.org/abs/1505.05424, section 3.4.
num_training_examples = 1000
loss += bbb.get_total_kl_cost() / num_training_examples
optimizer = tf.train.GradientDescentOptimizer(1e-3)
train_op = optimizer.minimize(loss)
```

## Reusing variables (e.g. RNNs).

A unique `tf.Variable` will only count once towards the KL cost returned by
`get_total_kl_cost()` regardless of how many times it is used in the graph.
By default, every time a variable is retrieved by `tf.get_variable()`,
new sampling noise will be used. To disable this behavior, pass the argument
`fresh_noise_per_connection=False` to the `bayes_by_backprop_getter` factory.

If using `tf.while_loop`, noise is *not* resampled per iteration regardless of
the value of `fresh_noise_per_connection`. This is because tensors created
outside a `tf.while_loop` are evaluated only once. You can disable this
behaviour by passing the argument `keep_control_dependencies=True` to the
`bayes_by_backprop_getter` factory.

## Contact
jmenick@
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import weakref

import tensorflow as tf
import tensorflow_probability as tfp

_DEFAULT_SCALE_TRANSFORM = tf.nn.softplus
_OK_DTYPES_FOR_BBB = (tf.float16, tf.float32, tf.float64, tf.bfloat16)
_OK_PZATION_TYPE = tfp.distributions.FULLY_REPARAMETERIZED


class _WeakRegistry(weakref.WeakKeyDictionary):

  def __getitem__(self, key):
    try:
      return weakref.WeakKeyDictionary.__getitem__(self, key)
    except KeyError:
      new_value = collections.OrderedDict()
      self[key] = new_value
      return new_value

_all_var_metadata_registry = _WeakRegistry()


def inverse_softplus(y):
  """The inverse of the softplus function.

  Computes the *inverse* of softplus, a function which maps an
  unconstrained real number to the positive reals, e.g. to squash an
  unconstrained neural network activation to parameterize a variance.

  Args:
    y: A positive number.
  Returns:
    The number `x` such that softplus(x) = y.
  """
  return math.log(math.exp(y) - 1.0)


def scale_variable_initializer(desired_scale):
  return tf.constant_initializer(inverse_softplus(desired_scale))


# pylint: disable=old-style-class
class EstimatorModes:
  sample = "sample"
  mean = "mean"
  last_sample = "last"
# pylint: enable=old-style-class


_VariableMetadata = collections.namedtuple(
    "VariableMetadata",
    ["raw_variable_name", "raw_variable_shape", "scope_name", "posterior",
     "posterior_estimate", "prior", "kl_cost", "prior_vars", "posterior_vars"])


# pylint: disable=keyword-arg-before-vararg
def diagonal_gaussian_posterior_builder(
    getter, name, shape=None, *args, **kwargs):
  """A pre-canned builder for diagonal gaussian posterior distributions.

  Given a true `getter` function and arguments forwarded from `tf.get_variable`,
  return a distribution object for a diagonal posterior over a variable of the
  requisite shape.

  Args:
    getter: The `getter` passed to a `custom_getter`. Please see the
      documentation for `tf.get_variable`.
    name: The `name` argument passed to `tf.get_variable`.
    shape: The `shape` argument passed to `tf.get_variable`.
    *args: See positional arguments passed to `tf.get_variable`.
    **kwargs: See keyword arguments passed to `tf.get_variable`.

  Returns:
    An instance of `tfp.distributions.Normal` representing the posterior
    distribution over the variable in question.
  """
  # Please see the documentation for
  # `tfp.distributions.param_static_shapes`.
  parameter_shapes = tfp.distributions.Normal.param_static_shapes(shape)

  loc_var = getter(
      name + "/posterior_loc", shape=parameter_shapes["loc"], *args, **kwargs)
  scale_var = getter(
      name + "/posterior_scale",
      shape=parameter_shapes["scale"],
      *args,
      **kwargs)
  posterior = tfp.distributions.Normal(
      loc=loc_var,
      scale=tf.nn.softplus(scale_var),
      name="{}_posterior_dist".format(name))
  return posterior
# pylint: enable=keyword-arg-before-vararg


# pylint: disable=keyword-arg-before-vararg
def fixed_gaussian_prior_builder(
    getter, name, dtype=None, *args, **kwargs):
  """A pre-canned builder for fixed gaussian prior distributions.

  Given a true `getter` function and arguments forwarded from `tf.get_variable`,
  return a distribution object for a scalar-valued fixed gaussian prior which
  will be broadcast over a variable of the requisite shape.

  Args:
    getter: The `getter` passed to a `custom_getter`. Please see the
      documentation for `tf.get_variable`.
    name: The `name` argument passed to `tf.get_variable`.
    dtype: The `dtype` argument passed to `tf.get_variable`.
    *args: See positional arguments passed to `tf.get_variable`.
    **kwargs: See keyword arguments passed to `tf.get_variable`.

  Returns:
    An instance of `tfp.distributions.Normal` representing the prior
    distribution over the variable in question.
  """
  del getter  # Unused.
  del args  # Unused.
  del kwargs  # Unused.
  loc = tf.constant(0.0, shape=(), dtype=dtype)
  scale = tf.constant(0.01, shape=(), dtype=dtype)
  return tfp.distributions.Normal(
      loc=loc, scale=scale, name="{}_prior_dist".format(name))
# pylint: enable=keyword-arg-before-vararg


def adaptive_gaussian_prior_builder(
    getter, name, *args, **kwargs):
  """A pre-canned builder for adaptive scalar gaussian prior distributions.

  Given a true `getter` function and arguments forwarded from `tf.get_variable`,
  return a distribution object for a scalar-valued adaptive gaussian prior
  which will be broadcast over a variable of the requisite shape. This prior's
  parameters (e.g `loc` and `scale` for a gaussian) will consist of a single
  learned scalar for the entire `tf.Variable` for which it serves as the prior,
  regardless of that `tf.Variable`'s shape.

  Args:
    getter: The `getter` passed to a `custom_getter`. Please see the
      documentation for `tf.get_variable`.
    name: The `name` argument passed to `tf.get_variable`.
    *args: See positional arguments passed to `tf.get_variable`.
    **kwargs: See keyword arguments passed to `tf.get_variable`.

  Returns:
    An instance of `tfp.distributions.Normal` representing the prior
    distribution over the variable in question.
  """
  kwargs["shape"] = ()
  loc_var = getter(name + "_prior_loc", *args, **kwargs)
  kwargs["initializer"] = scale_variable_initializer(0.01)
  scale_var = getter(name + "_prior_scale", *args, **kwargs)
  prior = tfp.distributions.Normal(
      loc=loc_var, scale=tf.nn.softplus(scale_var),
      name="{}_prior_dist".format(name))
  return prior


def stochastic_kl_builder(posterior, prior, sample):
  """A pre-canned builder for a ubiquitous stochastic KL estimator."""
  return tf.subtract(
      tf.reduce_sum(posterior.log_prob(sample)),
      tf.reduce_sum(prior.log_prob(sample)))


def analytic_kl_builder(posterior, prior, sample):
  """A pre-canned builder for the analytic kl divergence."""
  del sample
  return tf.reduce_sum(tfp.distributions.kl_divergence(posterior, prior))


def bayes_by_backprop_getter(
    posterior_builder=diagonal_gaussian_posterior_builder,
    prior_builder=fixed_gaussian_prior_builder,
    kl_builder=stochastic_kl_builder,
    sampling_mode_tensor=None,
    fresh_noise_per_connection=True,
    keep_control_dependencies=False):
  """Creates a custom getter which does Bayes by Backprop.

  Please see `tf.get_variable` for general documentation on custom getters.

  All arguments are optional. If nothing is configued, then a diagonal gaussian
  posterior will be used, and a fixed N(0, 0.01) prior will be used. Please
  see the default `posterior_builder` and `prior_builder` for a more detailed
  understanding of the default settings.

  Args:
    posterior_builder: A builder function which constructs an instance of
      `tfp.distributions.Distribution` which shall serve as the posterior over
      the `tf.Variable` of interest. The builder receives the `getter` and the
      arguments forwarded from `tf.get_variable`. Suppose one wrote

      ```
      tf.get_variable(
          'weights', shape=(3,), initializer=tf.zeros_initializer,
          dtype=tf.float32)
      ```

      then the `posterior_builder` argument would receive the `name`, `shape`,
      `initializer`, and `dtype` arguments passed above. The builder must return
      a `tfp.distributions.Distribution` object.

      Please see the `tf.get_variable` for documentation on `custom_getter` and
      `getter`, and see `bbb.diagonal_gaussian_posterior_builder`
      (the default) for an example of using this builder API.
    prior_builder: A builder function which constructs an instance of
      `tfp.distributions.Distribution` which shall serve as the prior over the
      `tf.Variable` of interest. Identical API to `posterior_builder`. See
      `bbb.fixed_gaussian_prior_builder` (the default) for an example.
    kl_builder: A builder function which receives the posterior distribution,
      prior distribution, and a sample from the posterior. It returns a
      scalar-shaped `tf.Tensor` representing the total KL cost for the
      `tf.Variable` in question. See `bbb.stochastic_kl_builder` (default) and
      `bbb.analytic_kl_builder` for examples.
    sampling_mode_tensor: A `tf.Tensor` which determines how an estimate from
      the posterior is produced. It must be scalar-shaped and have a `dtype` of
      `tf.string`. Valid values for this tensor are `bbb.EstimatorModes.sample`
      (which is the default), `bbb.EstimatorModes.mean`, and
      `bbb.EstimatorModes.last_sample`. `bbb.EstimatorModes.sample` is
      appropriate for training, and `bbb.EstimatorModes.mean` can be used
      at test time.
    fresh_noise_per_connection: A boolean. Indicates that each time a stochastic
      variable is retrieved with this custom getter, new sampling noise should
      be used. This is `True` by default. If this argument is set to `False`,
      then the same noise is used for each connection. Note that this does not
      apply to connections within a `tf.while_loop`; the same sampling noise
      is always used in different iterations of a `tf.while_loop` within one
      `session.run()` call. See the unit tests for details.
    keep_control_dependencies: A boolean. This argument should only be
      used by advanced users. Indicates that each time a stochastic variable is
      retrieved in the loop body of a `tf.while_loop` construct, new sampling
      noise should be used.
      The default behavior is `False`, so that RNNs use the same weights at each
      recurrent time step. This is done by removing the creation of the Variable
      from any existing control flow contexts. Notably, the Variables will be
      created outside the context of any tf.while_loop, making them fetchable.
      When this argument is `True`, any Variables used in the loop body of a
      `tf.while_loop` will be non-fetchable. If the KL cost needs to be
      evaluated, the Variable must *first* be used *outside* the loop body. This
      op using the Variable simply needs to be placed on the graph to get a
      stochastic estimate of the KL; it doesn't need to ever be used. Example:

      ```
      def loop_body(i):
        logits = sonnet_module(queue)
        i = i + 1

      with tf.variable_scope('bbb', custom_getter=bbb.bayes_by_backprop_getter(
          fresh_noise_per_connection=True,
          keep_control_dependencies=True)):
        unused_op = sonnet_module(queue)  # Adds KL estimate to bbb Collection
        final_i = tf.while_loop(lambda i: i < 5, loop_body, tf.constant(0.))
      ```

      Here when we add `unused_op` to the graph, we also add a number of tensors
      associated with the particular stochastic variable, including its
      contribution to the KL cost, to a graph-level registry. These are
      organized in a per-stochastic-variable data structure and be accessed with
      `bbb.get_variable_metadata()`. Without this line, these Tensors would
      instead be added the first time the Variable is used in the while_loop,
      which would make them non-fetchable.

      In all cases, the KL cost is only added once per Variable, which is the
      correct behavior, since if a variable is used multiple times in a model,
      the KL cost should remain unaffected.

  Returns:
    A `custom_getter` function which implements Bayes by Backprop.
  """

  if sampling_mode_tensor is None:
    sampling_mode_tensor = tf.constant(EstimatorModes.sample)

  def custom_getter(getter, name, *args, **kwargs):
    """The custom getter that will be returned."""
    if kwargs.get("trainable") is False:
      return getter(name, *args, **kwargs)
    if kwargs["dtype"] not in _OK_DTYPES_FOR_BBB:
      raise ValueError("Disallowed data type {}.".format(kwargs["dtype"]))

    var_scope = tf.get_variable_scope()
    if var_scope.reuse and not fresh_noise_per_connection:
      # Re-use the sampling noise by returning the very same posterior sample
      # if configured to do so.
      the_match = [
          x for x in get_variable_metadata() if x.raw_variable_name == name]
      if not the_match:
        raise ValueError(
            "Internal error. No metadata for variable {}".format(name))
      if len(the_match) > 1:
        raise ValueError(
            "Multiple matches for variable {}. Matches: {}".format(
                name, [x.raw_variable_name for x in the_match]))

      return the_match[0].posterior_estimate

    raw_variable_shape = kwargs["shape"]

    def construct_subgraph():
      """Constructs subgraph used to reparameterize the variable in question."""
      posterior = posterior_builder(
          getter,
          name=name,
          *args, **kwargs)
      prior = prior_builder(
          getter,
          name=name,
          *args, **kwargs)

      # If the user does not return an extra dictionary of prior variables,
      # then fill in an empty dictionary.
      try:
        posterior_dist, posterior_vars = posterior
      except TypeError:
        posterior_dist, posterior_vars = posterior, {}

      try:
        prior_dist, prior_vars = prior
      except TypeError:
        prior_dist, prior_vars = prior, {}

      if posterior_dist.reparameterization_type != _OK_PZATION_TYPE:
        raise ValueError(
            "Distribution {} incompatible with Bayes by Backprop.".format(
                posterior_dist.__class__.__name__))

      posterior_estimator = _produce_posterior_estimate(posterior_dist,
                                                        sampling_mode_tensor,
                                                        name)
      kl_cost = kl_builder(posterior_dist, prior_dist, posterior_estimator)
      variable_metadata = _VariableMetadata(
          raw_variable_name=name,
          raw_variable_shape=raw_variable_shape,
          scope_name=var_scope.name,
          posterior=posterior_dist,
          posterior_estimate=posterior_estimator,
          prior=prior_dist,
          kl_cost=kl_cost,
          prior_vars=prior_vars,
          posterior_vars=posterior_vars)
      return posterior_estimator, variable_metadata

    # Entering the `tf.control_dependencies(None)` context is crucial to
    # provide compatibility with `tf.while_loop` and thus RNNs. The main thing
    # it does is making the `kl_cost` fetchable by causing these ops to be
    # created outside the context of any tf.while_loop. Note also that it causes
    # a RNN core's weights to be sampled just once when unrolled over a
    # sequence, rather than at every timestep.
    control_deps = [] if keep_control_dependencies else None
    with tf.control_dependencies(control_deps):
      posterior_estimator, var_metadata = construct_subgraph()

    # Only add these ops to a collection once per unique variable.
    # This is to ensure that KL costs are not tallied up more than once.
    var_with_name = _all_var_metadata_registry[tf.get_default_graph()].get(name)
    if var_with_name is None:
      _all_var_metadata_registry[tf.get_default_graph()][name] = var_metadata

    return posterior_estimator
  return custom_getter


def _produce_posterior_estimate(posterior_dist, posterior_estimate_mode,
                                raw_var_name):
  """Create tensor representing estimate of posterior.

  Args:
    posterior_dist: An instance of `tfp.distributions.Distribution`.
        The variational posterior from which to produce an estimate of the
        variable in question.
    posterior_estimate_mode: A `Tensor` of dtype `tf.string`, which
        determines the inference mode.
    raw_var_name: The name of the variable over which inference is done.

  Returns:
    `z_sample`, a `Tensor` representing an estimate derived from the
        posterior distribution.
  """
  conds = [
      tf.equal(posterior_estimate_mode,
               tf.constant(EstimatorModes.sample),
               name="equal_sample_mode"),
      tf.equal(posterior_estimate_mode,
               tf.constant(EstimatorModes.mean),
               name="equal_mean_mode"),
      tf.equal(posterior_estimate_mode,
               tf.constant(EstimatorModes.last_sample),
               name="equal_last_sample_mode"),
  ]
  # pylint: disable=unnecessary-lambda
  results = [
      lambda: posterior_dist.sample(),
      lambda: posterior_dist.mean(),
      lambda: posterior_dist.last_sample()
  ]

  def default_case_branch_raising_error():
    err_msg = "Invalid posterior estimate mode."
    raise_err = tf.Assert(tf.constant(False), data=[tf.constant(err_msg)])
    with tf.control_dependencies([raise_err]):
      return posterior_dist.mean()

  if hasattr(posterior_dist, "last_sample"):
    cases = {conds[0]: results[0], conds[1]: results[1], conds[2]: results[2]}
  else:
    cases = {conds[0]: results[0], conds[1]: results[1]}
  z_sample = tf.case(
      cases,
      exclusive=True,
      default=default_case_branch_raising_error,
      name="{}_posterior_estimate".format(raw_var_name))
  # pylint: enable=unnecessary-lambda
  return z_sample


def get_total_kl_cost(name="total_kl_cost", filter_by_name_substring=None):
  """Get the total cost for all (or a subset of) the stochastic variables.

  Args:
    name: A name for the tensor representing the total kl cost.
    filter_by_name_substring: A string used to filter which variables count
      toward the total KL cost. By default, this argument is `None`, and all
      variables trained using Bayes by Backprop are included. If this argument
      is provided, the variables whose KL costs are summed will be all those
      whose name contains `filter_by_name_substring`. An example use of this
      would be to select all variables within a particular scope.

  Returns:
    A tensor representing the total KL cost in the ELBO loss.
  """
  all_variable_metadata = get_variable_metadata(filter_by_name_substring)
  if not all_variable_metadata:
    tf.logging.warning("No Bayes by Backprop variables found!")
    return tf.constant(0.0, shape=())
  return tf.add_n([md.kl_cost for md in all_variable_metadata], name=name)


def get_variable_metadata(scope_name_substring=None):
  variable_metadata = _all_var_metadata_registry[tf.get_default_graph()]
  all_variable_metadata = variable_metadata.values()
  if scope_name_substring is not None:
    all_variable_metadata = [x for x in all_variable_metadata
                             if scope_name_substring in x.scope_name]
  else:
    # Ensure all_variable_metadata is always a list.
    all_variable_metadata = list(all_variable_metadata)
  return all_variable_metadata
