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

"""Wrappers to add residual and skip connections to Sonnet modules."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from sonnet.python.modules import base
from sonnet.python.modules import rnn_core
import tensorflow as tf

from tensorflow.python.util import nest


class Residual(base.AbstractModule):
  """Adds a residual connection to a base module.

  This module wraps a module M, where if M with traditionally output M(X),
  Residual(M)(x) = M(x) + x.
  """

  def __init__(self, base_module, name="residual"):
    super(Residual, self).__init__(name=name)

    self._base_module = base_module

  def _build(self, inputs, **kwargs):
    outputs = self._base_module(inputs, **kwargs)
    residual = nest.map_structure(lambda inp, out: inp + out, inputs, outputs)
    return residual


class ResidualCore(rnn_core.RNNCore):
  """Adds a residual connection to a base RNN core.

  This module wraps a module M, where if M with traditionally output M(X),
  Residual(M)(x) = M(x) + x.
  """

  def __init__(self, base_core, name="residual_core"):
    super(ResidualCore, self).__init__(name=name)
    self._base_core = base_core

  def _build(self, inputs, prev_state, **kwargs):
    outputs, new_state = self._base_core(inputs, prev_state, **kwargs)
    residual = nest.map_structure(lambda inp, out: inp + out, inputs, outputs)
    return residual, new_state

  @property
  def output_size(self):
    return self._base_core.output_size

  @property
  def state_size(self):
    return self._base_core.state_size


class SkipConnectionCore(rnn_core.RNNCore):
  """Adds a skip connection to the base RNN core.

  This concatenates the input to the output of the base core.
  """

  def __init__(self, base_core, input_shape=None, name="skip_connection_core"):
    """Construct a SkipConnectionCore.

    Args:
      base_core: Base RNNCore to wrap.
      input_shape: Shape of the input as tuple, excluding the batch size.
      name: Name of the module.
    """
    super(SkipConnectionCore, self).__init__(name=name)
    self._base_core = base_core
    self._input_shape = input_shape

  def _build(self, inputs, prev_state, **kwargs):
    if not self._input_shape:
      self._input_shape = inputs.get_shape()[1:]
    outputs, new_state = self._base_core(inputs, prev_state, **kwargs)

    outputs = nest.map_structure(lambda inp, out: tf.concat((inp, out), -1),
                                 inputs, outputs)

    return outputs, new_state

  @property
  def output_size(self):
    if not self._input_shape:
      raise ValueError(
          "Output size unknown. You must provide the input_shape to the class' "
          "constructor or connect the module into the graph."
      )

    leading_dims = tuple(self._input_shape[:-1])
    final_input_dim = self._input_shape[-1]

    return tf.TensorShape(leading_dims +
                          (self._base_core.output_size[-1] + final_input_dim,))

  @property
  def state_size(self):
    return self._base_core.state_size
