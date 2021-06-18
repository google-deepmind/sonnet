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
"""Exponential moving average for Sonnet."""

from typing import Optional

from sonnet.src import metrics
from sonnet.src import once
from sonnet.src import types
import tensorflow as tf


class ExponentialMovingAverage(metrics.Metric):
  """Maintains an exponential moving average for a value.

  Note this module uses debiasing by default. If you don't want this please use
  an alternative implementation.

  This module keeps track of a hidden exponential moving average that is
  initialized as a vector of zeros which is then normalized to give the average.
  This gives us a moving average which isn't biased towards either zero or the
  initial value. Reference (https://arxiv.org/pdf/1412.6980.pdf)

  Initially:

      hidden_0 = 0

  Then iteratively:

      hidden_i = (hidden_{i-1} - value) * (1 - decay)
      average_i = hidden_i / (1 - decay^i)

  Attributes:
    average: Variable holding average. Note that this is None until the first
      value is passed.
  """

  def __init__(self, decay: types.FloatLike, name: Optional[str] = None):
    """Creates a debiased moving average module.

    Args:
      decay: The decay to use. Note values close to 1 result in a slow decay
        whereas values close to 0 result in faster decay, tracking the input
        values more closely.
      name: Name of the module.
    """
    super().__init__(name=name)
    self._decay = decay
    self._counter = tf.Variable(
        0, trainable=False, dtype=tf.int64, name="counter")

    self._hidden = None
    self.average = None

  def update(self, value: tf.Tensor):
    """Applies EMA to the value given."""
    self.initialize(value)

    self._counter.assign_add(1)
    value = tf.convert_to_tensor(value)
    counter = tf.cast(self._counter, value.dtype)
    self._hidden.assign_sub((self._hidden - value) * (1 - self._decay))
    self.average.assign((self._hidden / (1. - tf.pow(self._decay, counter))))

  @property
  def value(self) -> tf.Tensor:
    """Returns the current EMA."""
    return self.average.read_value()

  def reset(self):
    """Resets the EMA."""
    self._counter.assign(tf.zeros_like(self._counter))
    self._hidden.assign(tf.zeros_like(self._hidden))
    self.average.assign(tf.zeros_like(self.average))

  @once.once
  def initialize(self, value: tf.Tensor):
    self._hidden = tf.Variable(
        tf.zeros_like(value), trainable=False, name="hidden")
    self.average = tf.Variable(
        tf.zeros_like(value), trainable=False, name="average")
