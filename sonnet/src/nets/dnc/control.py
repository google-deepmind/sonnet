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
"""DNC Control Modules.

These modules receive input and output parameters for the memory access module.
We also alias external controllers in this module that are relevant, so they can
be specified by string name in the core config.
"""

import sys

from sonnet.src import linear
from sonnet.src import recurrent
import tensorflow as tf


def get_controller_ctor(controller_name):
  """Returns the constructor for a givn controller name."""
  if controller_name == 'LSTM':
    return recurrent.LSTM
  elif controller_name == 'GRU':
    return recurrent.GRU
  else:
    # References for other controllers can be added here
    return getattr(sys.modules[__name__], controller_name)


class FeedForward(recurrent.RNNCore):
  """FeedForward controller module.


  Single feedforward linear layer, wrapped as an RNN core for convenience. There
  is no computation performed on the state.

      y <- activation(linear(x))
      s_t+1 <- s_t
  """

  def __init__(self,
               hidden_size,
               activation=tf.nn.tanh,
               dtype=tf.float32,
               name=None):
    """Initializes the FeedForward Module.

    Args:
      hidden_size: number of hidden units in linear layer.
      activation: op for output activations.
      dtype: datatype of inputs to accept, defaults to tf.float32.
      name: module name (default 'feed_forward').
    """
    super().__init__(name=name)
    self.linear = linear.Linear(hidden_size)
    self.dtype = dtype
    self._activation = activation

  def __call__(self, inputs, prev_state):
    """Connects the FeedForward controller to the graph.

    Args:
      inputs: 2D Tensor [batch_size, input_size] input_size needs to be
        specified at construction time.
      prev_state: dummy state, 2D tensor of size [batch_size, 1]

    Returns:
      output: 2D Tensor [batch_size, hidden_size].
      next_state: the same dummy state passed in as an argument.
    """
    output = self.linear(inputs)
    if self._activation is not None:
      output = self._activation(output)
    return output, prev_state

  def initial_state(self, batch_size):
    return tf.zeros([batch_size, 1], dtype=self.dtype)


def deep_core(control_name,
              control_config,
              num_layers=1,
              skip_connections=True,
              name=None):
  """Constructs a deep control module.

  Args:
    control_name: Name of control module (e.g. "LSTM").
    control_config: Dictionary containing the configuration for the modules.
    num_layers: Number of layers.
    skip_connections: Boolean that indicates whether to use skip connections.
      See documenation for sonnet.DeepRnn in
      //learning/deepmind/tensorflow/sonnet/python/modules/basic_rnn.py for more
      information.
    name: module name.

  Returns:
    Deep control module.
  """
  control_class = get_controller_ctor(control_name)
  cores = [
      control_class(name='{}_{}'.format(control_name, i), **control_config)
      for i in range(num_layers)
  ]
  if skip_connections:
    return recurrent.deep_rnn_with_skip_connections(cores, name=name)
  else:
    return recurrent.DeepRNN(cores, name=name)
