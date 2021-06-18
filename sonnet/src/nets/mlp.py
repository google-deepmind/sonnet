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
"""A minimal interface mlp module."""

from typing import Callable, Iterable, Optional

from sonnet.src import base
from sonnet.src import initializers
from sonnet.src import linear
import tensorflow as tf


class MLP(base.Module):
  """A multi-layer perceptron module."""

  def __init__(self,
               output_sizes: Iterable[int],
               w_init: Optional[initializers.Initializer] = None,
               b_init: Optional[initializers.Initializer] = None,
               with_bias: bool = True,
               activation: Callable[[tf.Tensor], tf.Tensor] = tf.nn.relu,
               dropout_rate=None,
               activate_final: bool = False,
               name: Optional[str] = None):
    """Constructs an MLP.

    Args:
      output_sizes: Sequence of layer sizes.
      w_init: Initializer for Linear weights.
      b_init: Initializer for Linear bias. Must be `None` if `with_bias` is
        `False`.
      with_bias: Whether or not to apply a bias in each layer.
      activation: Activation function to apply between linear layers. Defaults
        to ReLU.
      dropout_rate: Dropout rate to apply, a rate of `None` (the default) or `0`
        means no dropout will be applied.
      activate_final: Whether or not to activate the final layer of the MLP.
      name: Optional name for this module.

    Raises:
      ValueError: If with_bias is False and b_init is not None.
    """
    if not with_bias and b_init is not None:
      raise ValueError("When with_bias=False b_init must not be set.")

    super().__init__(name=name)
    self._with_bias = with_bias
    self._w_init = w_init
    self._b_init = b_init
    self._activation = activation
    self._activate_final = activate_final
    self._dropout_rate = dropout_rate
    self._layers = []
    for index, output_size in enumerate(output_sizes):
      self._layers.append(
          linear.Linear(
              output_size=output_size,
              w_init=w_init,
              b_init=b_init,
              with_bias=with_bias,
              name="linear_%d" % index))

  def __call__(self, inputs: tf.Tensor, is_training=None) -> tf.Tensor:
    """Connects the module to some inputs.

    Args:
      inputs: A Tensor of shape `[batch_size, input_size]`.
      is_training: A bool indicating if we are currently training. Defaults to
        `None`. Required if using dropout.

    Returns:
      output: The output of the model of size `[batch_size, output_size]`.
    """
    use_dropout = self._dropout_rate not in (None, 0)
    if use_dropout and is_training is None:
      raise ValueError(
          "The `is_training` argument is required when dropout is used.")
    elif not use_dropout and is_training is not None:
      raise ValueError(
          "The `is_training` argument should only be used with dropout.")

    num_layers = len(self._layers)

    for i, layer in enumerate(self._layers):
      inputs = layer(inputs)
      if i < (num_layers - 1) or self._activate_final:
        # Only perform dropout if we are activating the output.
        if use_dropout and is_training:
          inputs = tf.nn.dropout(inputs, rate=self._dropout_rate)
        inputs = self._activation(inputs)

    return inputs

  def reverse(self,
              activate_final: Optional[bool] = None,
              name: Optional[str] = None) -> "MLP":
    """Returns a new MLP which is the layer-wise reverse of this MLP.

    NOTE: Since computing the reverse of an MLP requires knowing the input size
    of each linear layer this method will fail if the module has not been called
    at least once. See `snt.Deferred` as a possible solution to this problem.

    The contract of reverse is that the reversed module will accept the output
    of the parent module as input and produce an output which is the input size
    of the parent.

    >>> mlp = snt.nets.MLP([1, 2, 3])
    >>> y = mlp(tf.ones([1, 2]))
    >>> rev = mlp.reverse()
    >>> rev(y)
    <tf.Tensor: shape=(1, 2), ...>

    Args:
      activate_final: Whether the final layer of the MLP should be activated.
      name: Optional name for the new module. The default name will be the name
        of the current module prefixed with ``"reversed_"``.

    Returns:
      An MLP instance which is the reverse of the current instance. Note these
      instances do not share weights and, apart from being symmetric to each
      other, are not coupled in any way.
    """

    if activate_final is None:
      activate_final = self._activate_final
    if name is None:
      name = self.name + "_reversed"

    return MLP(
        output_sizes=(layer.input_size for layer in reversed(self.submodules)),
        w_init=self._w_init,
        b_init=self._b_init,
        with_bias=self._with_bias,
        activation=self._activation,
        dropout_rate=self._dropout_rate,
        activate_final=activate_final,
        name=name)
