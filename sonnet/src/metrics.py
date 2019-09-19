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
"""Base class for general metrics within Sonnet."""

from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

import abc
import six

from sonnet.src import base
from sonnet.src import once
import tensorflow as tf
from typing import Optional, Text


@six.add_metaclass(abc.ABCMeta)
class Metric(base.Module):
  """Metric base class."""

  @abc.abstractmethod
  def initialize(self, value):
    """Creates any input dependent variables or state."""

  @abc.abstractmethod
  def update(self, value):
    """Accumulates values."""

  @abc.abstractproperty
  def value(self):
    """Returns the current value of the metric."""

  @abc.abstractmethod
  def reset(self):
    """Resets the metric."""

  def __call__(self, value):
    """Updates the metric and returns the new value."""
    self.update(value)
    return self.value


class Sum(Metric):
  """Calculates the element-wise sum of the given values."""

  def __init__(self, name: Optional[Text] = None):
    super(Sum, self).__init__(name=name)
    self.sum = None

  @once.once
  def initialize(self, value: tf.Tensor):
    """See base class."""
    self.sum = tf.Variable(tf.zeros_like(value), trainable=False, name="sum")

  def update(self, value: tf.Tensor):
    """See base class."""
    self.initialize(value)
    self.sum.assign_add(value)

  @property
  def value(self) -> tf.Tensor:
    """See base class."""
    return tf.convert_to_tensor(self.sum)

  def reset(self):
    """See base class."""
    self.sum.assign(tf.zeros_like(self.sum))


class Mean(Metric):
  """Calculates the element-wise mean of the given values."""

  def __init__(self, name: Optional[Text] = None):
    super(Mean, self).__init__(name=name)
    self.sum = None
    self.count = tf.Variable(0, dtype=tf.int64, trainable=False, name="count")

  @once.once
  def initialize(self, value: tf.Tensor):
    """See base class."""
    self.sum = tf.Variable(tf.zeros_like(value), trainable=False, name="sum")

  def update(self, value: tf.Tensor):
    """See base class."""
    self.initialize(value)
    self.sum.assign_add(value)
    self.count.assign_add(1)

  @property
  def value(self) -> tf.Tensor:
    """See base class."""
    # TODO(cjfj): Assert summed type is floating-point?
    return self.sum / tf.cast(self.count, dtype=self.sum.dtype)

  def reset(self):
    self.sum.assign(tf.zeros_like(self.sum))
    self.count.assign(0)
