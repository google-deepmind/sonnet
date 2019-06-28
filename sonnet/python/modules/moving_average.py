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

"""Module that calculates a differentiable decaying moving average."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from sonnet.python.modules import base
import tensorflow as tf

from tensorflow.python.framework import function  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.training import moving_averages  # pylint: disable=g-direct-tensorflow-import


class MovingAverage(base.AbstractModule):
  """Calculates a differentiable decaying moving average.

  The moving average is kept in a variable that can either be local or global.
  The initial moving average value is set to the first value that is received
  by the module. The module lets gradients flow through the last element added
  to the moving average.
  """

  def __init__(self, decay=0.99, local=False, name="moving_average"):
    """Constructor.

    Args:
      decay: float in range [0, 1], decay of the moving average.
      local: bool, specifies whether the variables are local or not.
      name: string, name of the Sonnet module. Default is 'moving_average'.

    Raises:
      ValueError: if decay is not in the valid range [0, 1].
    """
    super(MovingAverage, self).__init__(name=name)
    if decay < 0.0 or decay > 1.0:
      raise ValueError("Decay must be a float in the [0, 1] range, "
                       "but is {}.".format(decay))
    self._decay = decay
    if local:
      self._collection = tf.GraphKeys.LOCAL_VARIABLES
    else:
      self._collection = tf.GraphKeys.GLOBAL_VARIABLES

  def reset(self):
    return tf.group(
        self._initialized.initializer,
        self._moving_average.initializer
    )

  def _build(self, inputs):
    """Returns the moving average of the values that went through `inputs`.

    Args:
      inputs: tensor.

    Returns:
      A moving average calculated as `(1 - decay) * inputs + decay * average`.
    """
    # This trivial op helps correct execution of control flow when inputs is
    # not a resource variable. See, for example,
    # MovingAverageTest.testAverage(use_resource_vars=False) in
    # moving_averate_test.py.
    # Note that inputs = tf.identity(inputs) does NOT have the same effect.
    inputs = 1 * inputs

    self._initialized = tf.get_variable(
        "initialized",
        shape=(),
        dtype=tf.bool,
        initializer=tf.constant_initializer(False),
        trainable=False,
        use_resource=True,
        collections=[self._collection])

    self._moving_average = tf.get_variable(
        "moving_average",
        shape=inputs.get_shape(),
        initializer=tf.zeros_initializer(),
        trainable=False,
        use_resource=True,
        collections=[self._collection])

    update_op = moving_averages.assign_moving_average(
        variable=self._moving_average,
        value=inputs,
        decay=self._decay,
        zero_debias=False,
        name="update_moving_average")

    def update():
      return tf.identity(update_op)

    def initialize():
      with tf.control_dependencies([update_op]):
        value = tf.assign(self._moving_average, inputs)
      with tf.control_dependencies([value]):
        update_initialized = tf.assign(self._initialized, True)
      with tf.control_dependencies([update_initialized]):
        value = tf.identity(value)
      return value

    moving_avg = tf.cond(self._initialized, update, initialize)
    return _pass_through_gradients(inputs, moving_avg)


def _pass_through_gradients(x, moving_avg, name="pass_through_gradients"):
  """Defines a custom backward pass, only differentiating through x.

  Returns an op returning the current value of the moving average in the forward
  pass, whilst allowing gradients to flow through the last entry to the moving
  average, operating in a similar fashion to
  ```
  x + tf.stop_gradient(moving_avg - x)
  ```
  but avoiding the related numerical issues.

  Args:
    x: the last entry to the moving average.
    moving_avg: the current value of the moving average.
    name: name for name scope of the pass through gradient operation.

  Returns:
    An op returning the current value of the moving average for the forward
       pass, allowing gradients to flow through the last op added to the moving
       average.
  """
  with tf.name_scope(name, "pass_through_gradients", values=[x, moving_avg]):
    x_dtype = x.dtype.base_dtype  # Convert ref dtypes to regular dtypes.
    moving_avg_dtype = moving_avg.dtype.base_dtype
    if x_dtype != moving_avg_dtype:
      raise TypeError(
          "Inputs to _differentiate_last_step are expected to be of the same "
          "type, but were {} and {}.".format(x_dtype, moving_avg_dtype))
    differentiate_last_step_op = _get_pass_through_gradients_op(x_dtype)
    output = differentiate_last_step_op(x, moving_avg)
    output.set_shape(x.get_shape())

  return output


# Implement simple memoization mechanism using a global dict.
_op_ctors = dict()


def _get_pass_through_gradients_op(dtype):
  """Creates an op switching between two ops for the forward and backward pass.

  This method produces a new op the first time it is called with a given `dtype`
  argument, and then uses the cached op each time it is called after that with
  the same `dtype`.

  Args:
    dtype: the dtype of the inputs.

  Returns:
    The switching op.
  """
  def _instantiate_op(dtype):
    """Instantiate pass through gradients op constructor for given dtype."""
    def _forward(x, moving_avg):
      del x
      return tf.identity(moving_avg)

    def _backward(op, grad):
      """Forwards the gradients moving the inputs within the valid range."""
      del op
      return grad, None

    func_name = "PassThroughGradients_{}".format(dtype.name)
    return function.Defun(
        dtype, dtype, python_grad_func=_backward,
        func_name=func_name)(_forward)

  if dtype.name not in _op_ctors:
    _op_ctors[dtype.name] = _instantiate_op(dtype)
  return _op_ctors[dtype.name]




