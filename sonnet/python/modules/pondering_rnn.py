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

"""Cores for RNNs with varying number of unrolls.

This file contains implementations for:
  * ACT (Adaptive Computation Time)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from sonnet.python.modules import basic
from sonnet.python.modules import rnn_core
from sonnet.python.ops import nest
import tensorflow as tf


def _nested_add(nested_a, nested_b):
  """Add two arbitrarily nested `Tensors`."""
  return nest.map(lambda a, b: a + b, nested_a, nested_b)


def _nested_unary_mul(nested_a, p):
  """Multiply `Tensors` in arbitrarily nested `Tensor` `nested_a` with `p`."""
  return nest.map(lambda a: p * a, nested_a)


def _nested_zeros_like(nested_a):
  return nest.map(tf.zeros_like, nested_a)


class ACTCore(rnn_core.RNNCore):
  """Adaptive computation time core.

  Implementation of the model described in "Adaptive Computation Time for
  Recurrent Neural Networks" paper, https://arxiv.org/abs/1603.08983.

  The `ACTCore` incorporates the pondering RNN of ACT, with different
  computation times for each element in the mini batch. Each pondering step is
  performed by the `core` passed to the constructor of `ACTCore`.

  The output of the `ACTCore` is made of `(act_out, (iteration, remainder)`,
  where

    * `iteration` counts the number of pondering step in each batch element;
    * `remainder` is the remainder as defined in the ACT paper;
    * `act_out` is the weighted average output of all pondering steps (see ACT
    paper for more info).
  """

  def __init__(self, core, output_size, threshold, get_state_for_halting,
               name="act_core"):
    """Constructor.

    Args:
      core: A `sonnet.RNNCore` object. This should only take a single `Tensor`
          in input, and output only a single flat `Tensor`.
      output_size: An integer. The size of each output in the sequence.
      threshold: A float between 0 and 1. Probability to reach for ACT to stop
          pondering.
      get_state_for_halting: A callable that can take the `core` state and
          return the input to the halting function.
      name: A string. The name of this module.

    Raises:
      ValueError: if `threshold` is not between 0 and 1.
      ValueError: if `core` has either nested outputs or outputs that are not
          one dimensional.
    """
    super(ACTCore, self).__init__(name=name)
    self._core = core
    self._output_size = output_size
    self._threshold = threshold
    self._get_state_for_halting = get_state_for_halting

    if not isinstance(self._core.output_size, tf.TensorShape):
      raise ValueError("Output of core should be single Tensor.")
    if self._core.output_size.ndims != 1:
      raise ValueError("Output of core should be 1D.")

    if not 0 <= self._threshold <= 1:
      raise ValueError("Threshold should be between 0 and 1, but found {}".
                       format(self._threshold))

  def initial_state(self, *args, **kwargs):
    return self._core.initial_state(*args, **kwargs)

  @property
  def output_size(self):
    return tf.TensorShape([self._output_size]), (tf.TensorShape([1]),
                                                 tf.TensorShape([1]))

  @property
  def state_size(self):
    return self._core.state_size

  @property
  def batch_size(self):
    self._ensure_is_connected()
    return self._batch_size

  @property
  def dtype(self):
    self._ensure_is_connected()
    return self._dtype

  def _cond(self, unused_x, unused_cumul_out, unused_prev_state,
            unused_cumul_state, cumul_halting, unused_iteration,
            unused_remainder):
    """The `cond` of the `tf.while_loop`."""
    return tf.reduce_any(cumul_halting < 1)

  def _body(self, x, cumul_out, prev_state, cumul_state,
            cumul_halting, iteration, remainder, halting_linear, x_ones):
    """The `body` of `tf.while_loop`."""
    # Increase iteration count only for those elements that are still running.
    all_ones = tf.constant(1, shape=(self._batch_size, 1), dtype=self._dtype)
    is_iteration_over = tf.equal(cumul_halting, all_ones)
    next_iteration = tf.where(is_iteration_over, iteration, iteration + 1)
    out, next_state = self._core(x, prev_state)
    # Get part of state used to compute halting values.
    halting_input = halting_linear(self._get_state_for_halting(next_state))
    halting = tf.sigmoid(halting_input, name="halting")
    next_cumul_halting_raw = cumul_halting + halting
    over_threshold = next_cumul_halting_raw > self._threshold
    next_cumul_halting = tf.where(over_threshold, all_ones,
                                  next_cumul_halting_raw)
    next_remainder = tf.where(over_threshold, remainder,
                              1 - next_cumul_halting_raw)
    p = next_cumul_halting - cumul_halting
    next_cumul_state = _nested_add(cumul_state,
                                   _nested_unary_mul(next_state, p))
    next_cumul_out = cumul_out + p * out

    return (x_ones, next_cumul_out, next_state, next_cumul_state,
            next_cumul_halting, next_iteration, next_remainder)

  def _build(self, x, prev_state):
    """Connects the core to the graph.

    Args:
      x: Input `Tensor` of shape `(batch_size, input_size)`.
      prev_state: Previous state. This could be a `Tensor`, or a tuple of
          `Tensor`s.

    Returns:
      The tuple `(output, state)` for this core.

    Raises:
      ValueError: if the `Tensor` `x` does not have rank 2.
    """
    x.get_shape().with_rank(2)
    self._batch_size = x.get_shape().as_list()[0]
    self._dtype = x.dtype

    x_zeros = tf.concat(
        [x, tf.zeros(
            shape=(self._batch_size, 1), dtype=self._dtype)], 1)
    x_ones = tf.concat(
        [x, tf.ones(
            shape=(self._batch_size, 1), dtype=self._dtype)], 1)
    # Weights for the halting signal
    halting_linear = basic.Linear(name="halting_linear", output_size=1)

    body = functools.partial(
        self._body, halting_linear=halting_linear, x_ones=x_ones)
    cumul_halting_init = tf.zeros(shape=(self._batch_size, 1),
                                  dtype=self._dtype)
    iteration_init = tf.zeros(shape=(self._batch_size, 1), dtype=self._dtype)
    core_output_size = [x.value for x in self._core.output_size]
    out_init = tf.zeros(shape=(self._batch_size,) + tuple(core_output_size),
                        dtype=self._dtype)
    cumul_state_init = _nested_zeros_like(prev_state)
    remainder_init = tf.zeros(shape=(self._batch_size, 1), dtype=self._dtype)
    (unused_final_x, final_out, unused_final_state, final_cumul_state,
     unused_final_halting, final_iteration, final_remainder) = tf.while_loop(
         self._cond, body, [x_zeros, out_init, prev_state, cumul_state_init,
                            cumul_halting_init, iteration_init, remainder_init])

    act_output = basic.Linear(
        name="act_output_linear", output_size=self._output_size)(final_out)

    return (act_output, (final_iteration, final_remainder)), final_cumul_state
