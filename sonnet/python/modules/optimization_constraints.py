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
# ===================================================

"""Tools for constrained optimization.

These classes and methods implement the logic described in:
Danilo Rezende and Fabio Viola, 'Taming VAEs': https://arxiv.org/abs/1810.00597
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numbers
# Dependency imports
import numpy as np

from sonnet.python.modules import basic
from sonnet.python.modules import scale_gradient

import tensorflow as tf

from tensorflow.python.framework import function  # pylint: disable=g-direct-tensorflow-import

LAGRANGE_MULTIPLIERS = 'Lagrange Multipliers'
_GEQ_OPS = ('Greater', 'GreaterEqual')
_LEQ_OPS = ('Less', 'LessEqual')


class OptimizationConstraints(object):
  """Container for optimization constraints.

  Users can add to an OptimizationConstraints instance multiple inequality
  constraints, either implicitly passing inequality ops, such as
  `optimization_constraints.add(x < y)`, or explicitly specifying the constraint
  type, as in `optimization_constraints.add_geq(x, y)`.
  Users can finally add the constraints to the TensorFlow graph calling
  `optimization_constraints()`; when doing so, Lagrange multipliers are
  automatically added to the graph, so that users can optimize them alongside
  other variables in the graph, using the same optimizer and `minimize()`.

  Example usage:
  ```
  regularization_loss = model.regularization_loss(data)
  reconstruction_error = model.reconstruction_error(data)
  avg_reconstruction_error = snt.MovingAverage()(reconstruction_error)
  constraints = snt.OptimizationConstraints()
  constraints.add(avg_reconstruction_error < reconstruction_threshold)
  loss = regularization_loss + constraints()
  # The following call actually performs an update step for
  # min_{theta} max_{lambda} (
  #     regularization_loss(theta) +
  #     lambda * (avg_reconstruction_error - reconstruction_threshold))
  # where theta are the model parameters and lambda are the Lagrange
  # multipliers.
  update = optimizer.minimize(loss)
  ```
  """

  def __init__(self, rate=1.0, valid_range=None):
    """Instantiates a container for optimization constraints.

    Args:
      rate: optional float, default 1.0. Default factor for Lagrange multiplier
          gradient scaling. Use there `rate` argument to scale the gradients of
          the Lagrange multipliers - note that this parameter has no effect when
          using optimisers such as Adam. This parameter can be overridden
          when adding constraints to the container.
      valid_range: optional tuple of length 2, default None. Default valid range
          for Lagrange multipliers. This parameter can be overridden when adding
          constraints to the container.
    """
    self._constraints = []
    self._lagrange_multipliers = []
    self._valid_ranges = []
    self._rate = rate
    self._valid_range = valid_range
    self._penalty = None

  @property
  def constraints(self):
    return self._constraints

  @property
  def lagrange_multipliers(self):
    return self._lagrange_multipliers

  def add(self, expression, rate=None, valid_range=None, initializer=None):
    """Add inequality constraint whose type depends on analysis of input op.

    Args:
      expression: op of type `Greater`, `GreaterEqual`, `Less` or `LessEqual`.
          Note that `GreaterEqual` and `LessEqual` are accepted only for
          convenience, and will result in the same behavior as `Greater` and
          `Less` respectively.
      rate: optional float, default None. Factor for Lagrange multiplier
          gradient scaling. Use there `rate` argument to scale the gradients of
          the Lagrange multipliers - note that this parameter has no effect when
          using optimisers such as Adam. This parameter overrides the defaults
          defined instantiating the container.
      valid_range: optional tuple of length 2, default None. Default valid
          range for Lagrange multipliers. This parameter overrides the defaults
          defined instantiating the container.
      initializer: optional tensorflow initializer, array or value to be used
          for the Lagrange multiplier initialization. By default Lagrange
          multiplier will be initialized to 1.0.

    Returns:
      Self.

    Raises:
      `TypeError`, when input expression op is not one of `Greater`,
      `GreaterEqual`, `Less`, `LessEqual`.
    """
    self._assert_is_not_connected()
    lhs = expression.op.inputs[0]
    rhs = expression.op.inputs[1]
    op_type = expression.op.type
    if op_type in _GEQ_OPS:
      self.add_geq(
          lhs, rhs, rate=rate, valid_range=valid_range, initializer=initializer)
    elif op_type in _LEQ_OPS:
      self.add_leq(
          lhs, rhs, rate=rate, valid_range=valid_range, initializer=initializer)
    else:
      raise TypeError(
          'add currently only supports parsing of the following ops: {}'.format(
              _GEQ_OPS + _LEQ_OPS))
    return self

  def add_leq(self, lhs, rhs=0.0, rate=None, valid_range=None,
              initializer=None):
    """Add a 'less than' inequality constraint.

    Args:
      lhs: left hand argument of inequality expression.
      rhs: reft hand argument of inequality expression, defaults to 0.0.
      rate: optional float, default None. Factor for Lagrange multiplier
          gradient scaling. Use there `rate` argument to scale the gradients of
          the Lagrange multipliers - note that this parameter has no effect when
          using optimisers such as Adam. This parameter overrides the defaults
          defined instantiating the container.
      valid_range: optional tuple of length 2, default None. Default valid
          range for Lagrange multipliers. This parameter overrides the defaults
          defined instantiating the container.
      initializer: optional tensorflow initializer, array or value to be used
          for the Lagrange multiplier initialization. By default Lagrange
          multiplier will be initialized to 1.0.

    Returns:
      Self.
    """
    self._assert_is_not_connected()
    constraint_op = lhs - rhs
    self._constraints.append(constraint_op)
    valid_range = valid_range or self._valid_range
    self._valid_ranges.append(valid_range)
    if rate is None:
      rate = self._rate
    lag_mul = get_lagrange_multiplier(
        shape=constraint_op.shape, rate=rate, initializer=initializer,
        valid_range=valid_range)
    self._lagrange_multipliers.append(lag_mul)
    return self

  def add_geq(self, lhs, rhs=0.0, rate=None, valid_range=None,
              initializer=None):
    """Add a 'greater than' inequality constraint.

    Args:
      lhs: left hand argument of inequality expression.
      rhs: reft hand argument of inequality expression, defaults to 0.0.
      rate: optional float, default None. Factor for Lagrange multiplier
          gradient scaling. Use there `rate` argument to scale the gradients of
          the Lagrange multipliers - note that this parameter has no effect when
          using optimisers such as Adam. This parameter overrides the defaults
          defined instantiating the container.
      valid_range: optional tuple of length 2, default None. Default valid
          range for Lagrange multipliers. This parameter overrides the defaults
          defined instantiating the container.
      initializer: optional tensorflow initializer, array or value to be used
          for the Lagrange multiplier initialization. By default Lagrange
          multiplier will be initialized to 1.0.

    Returns:
      Self.
    """
    self._assert_is_not_connected()
    constraint_op = rhs - lhs
    return self.add_leq(
        constraint_op, rate=rate, valid_range=valid_range,
        initializer=initializer)

  def __call__(self):
    """Adds constrains and Lagrange multipliers to graph."""
    if self._is_connected:
      return self._penalty

    self._penalty = tf.zeros(())
    for l, c in zip(self._lagrange_multipliers, self._constraints):
      self._penalty += tf.reduce_sum(l * c)
    return self._penalty

  @property
  def _is_connected(self):
    return self._penalty is not None

  def _assert_is_not_connected(self):
    if self._is_connected:
      raise ValueError(
          'Cannot add further constraints once OptimizationConstraints has '
          'been connected to the graph by calling it.')


def get_lagrange_multiplier(shape=(),
                            rate=1.0,
                            initializer=1.0,
                            maximize=True,
                            valid_range=None,
                            name='lagrange_multiplier'):
  """Lagrange multiplier factory.

  This factory returns ops that help setting up constrained optimization
  problems in Tensorflow. Given a constraint function op (either scalar or
  vectorial), use this function to instantiate a Lagrange multiplier op, then
  dot product the two and add them to the loss that is being optimized over.
  There is no need to instantiate a second optimizer to solve the minmax
  problem, as the Lagrange Multiplier op is setup to manipulate its own
  gradients so that a single optmizer can be used to update all the variables
  correctly.

  Args:
    shape: Lagrange multipliers can be used with both scalar and vector
        constraint functions; when using vector constraints use the shape kwarg
        to pass in shape information and instantiate variables of the correct
        shape.
    rate: Scalar used to scale the magnitude of gradients of the Lagrange
        multipliers, defaulting to 1e-2. Using the default value will make the
        Lagrange multipliers updates slower compared to the ones for the model's
        parameters.
    initializer: Initializer for the Lagrange multipliers. Note that
        when using inequality constraints the initial value of the multiplier
        will be transformed via the parametrization function.
    maximize: Boolean, True if we want to maximize the loss w.r.t. the Lagrange
        multipliers, False otherwise.
    valid_range: tuple, or list. of values used to clip the value of the
        (possibly reparametrized) Lagrange multipliers.
    name: Name of the Lagrange multiplier op.

  Returns:
    An op to be inserted in the graph, by multipling it with a constraint op
        and adding the resulting op to a loss. The Lagrange multiplier
        gradients are modified to that by calling minimize on the loss the
        optimizer will actually minimize w.r.t. to the model's parameters and
        maximize w.r.t. the Lagrande multipliers, hence enforcing the
        constraints.

  Raises:
    ValueError: If the Lagrange multiplier is set to enforce an equality
        constraint and a parametrization function is also provided.
  """
  initializer = initializer or np.ones(shape=shape)
  if isinstance(initializer, (numbers.Number, np.ndarray, list, tuple)):
    initializer = tf.constant_initializer(initializer)
  initializer = _LagrangeMultiplierInitializer(initializer)

  lambda_var = basic.TrainableVariable(
      name=name, shape=shape, initializers={'w': initializer})()
  tf.add_to_collection(LAGRANGE_MULTIPLIERS, lambda_var)

  lag_multiplier = _parametrize(lambda_var, rate=rate)
  lag_multiplier.set_shape(shape)
  if valid_range:
    lag_multiplier = _constrain_to_range(lag_multiplier, *valid_range)

  return lag_multiplier if maximize else -lag_multiplier


def _squared_softplus(x):
  return tf.nn.softplus(x) ** 2


def _inv_squared_softplus(x):
  return tf.contrib.distributions.softplus_inverse(tf.sqrt(x))


def _parametrize(x, rate=1.0):
  return scale_gradient.scale_gradient(_squared_softplus(x), -rate)


class _LagrangeMultiplierInitializer(object):
  """Initializer applying inv squared softplus to a user defined initializer."""

  def __init__(self, initializer, dtype=tf.float32):
    self.dtype = tf.as_dtype(dtype)
    self._initializer = initializer

  def __call__(self, shape, dtype=None, partition_info=None):
    initial_values = self._initializer(shape, dtype, partition_info)
    return _inv_squared_softplus(initial_values)

  def get_config(self):
    return {'dtype': self.dtype.name}


# Implement simple memoization mechanism using a global dict.
_op_ctors = dict()


def _get_constrain_to_range_op(dtype):
  """Creates an op that keep values within a given range using a Defun.

  This method produces a new op the first time it is called with a given `dtype`
  argument, and then uses the cached op each time it is called after that with
  the same `dtype`. The min and max valuee are given as arguments for the
  forward pass method so that it can be used in the backwards pass.

  Args:
    dtype: the dtype of the input whose values are clipped.

  Returns:
    The op that clips the values.
  """
  def _instantiate_op(dtype):
    """Instantiate constrain to range op constructor for given dtype."""
    def constrain_to_range_forward(x, clip_value_min, clip_value_max):
      return tf.clip_by_value(x, clip_value_min, clip_value_max)

    def constrain_to_range_backward(op, grad):
      """Forwards the gradients moving the inputs within the valid range."""
      x = op.inputs[0]
      clip_value_min = op.inputs[1]
      clip_value_max = op.inputs[2]
      zeros = tf.zeros_like(grad)

      condition = tf.logical_and(x < clip_value_min, grad < 0)
      grad = tf.where(condition, zeros, grad)

      condition = tf.logical_and(x > clip_value_max, grad > 0)
      grad = tf.where(condition, zeros, grad)
      return grad, None, None

    func_name = 'ConstrainToRange_{}'.format(dtype.name)
    return function.Defun(
        dtype, dtype, dtype, python_grad_func=constrain_to_range_backward,
        func_name=func_name)(constrain_to_range_forward)

  if dtype.name not in _op_ctors:
    _op_ctors[dtype.name] = _instantiate_op(dtype)
  return _op_ctors[dtype.name]


def _constrain_to_range(x, min_value, max_value, name='constrain_to_range'):
  """Clips the inputs to a given range, whilst forwarding gradients."""
  if not x.dtype.is_floating:
    raise ValueError('_clip_by_value does not support non-float `x` inputs.')

  with tf.name_scope(name, 'constrain_to_range', values=[x]):
    dtype = x.dtype.base_dtype  # Convert ref dtypes to regular dtypes.
    min_tensor = tf.convert_to_tensor(min_value, dtype=dtype)
    max_tensor = tf.convert_to_tensor(max_value, dtype=dtype)

    constrain_to_range_op = _get_constrain_to_range_op(dtype)
    output = constrain_to_range_op(x, min_tensor, max_tensor)
    output.set_shape(x.get_shape())

  return output


