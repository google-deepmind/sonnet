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

"""A minimal interface mlp module."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from six.moves import xrange  # pylint: disable=redefined-builtin
from sonnet.python.modules import base
from sonnet.python.modules import basic
from sonnet.python.modules import util

import tensorflow as tf
from tensorflow.python.layers import utils


class MLP(base.AbstractModule, base.Transposable):
  """A Multi-Layer perceptron module."""

  def __init__(self,
               output_sizes,
               activation=tf.nn.relu,
               activate_final=False,
               initializers=None,
               partitioners=None,
               regularizers=None,
               use_bias=True,
               use_dropout=False,
               custom_getter=None,
               name="mlp"):
    """Constructs an MLP module.

    Args:
      output_sizes: An iterable of output dimensionalities as defined in
        `basic.Linear`. Output size can be defined either as number or via a
        callable. In the latter case, since the function invocation is deferred
        to graph construction time, the user must only ensure that entries can
        be called when build is called. Each entry in the iterable defines
        properties in the corresponding linear layer.
      activation: An activation op. The activation is applied to intermediate
        layers, and optionally to the output of the final layer.
      activate_final: Boolean determining if the activation is applied to
        the output of the final layer. Default `False`.
      initializers: Optional dict containing ops to initialize the linear
        layers' weights (with key 'w') or biases (with key 'b').
      partitioners: Optional dict containing partitioners to partition the
        linear layers' weights (with key 'w') or biases (with key 'b').
      regularizers: Optional dict containing regularizers for the linear layers'
        weights (with key 'w') and the biases (with key 'b'). As a default, no
        regularizers are used. A regularizer should be a function that takes
        a single `Tensor` as an input and returns a scalar `Tensor` output, e.g.
        the L1 and L2 regularizers in `tf.contrib.layers`.
      use_bias: Whether to include bias parameters in the linear layers.
        Default `True`.
      use_dropout: Whether to perform dropout on the linear layers.
        Default `False`.
      custom_getter: Callable or dictionary of callables to use as
        custom getters inside the module. If a dictionary, the keys
        correspond to regexes to match variable names. See the `tf.get_variable`
        documentation for information about the custom_getter API.
      name: Name of the module.

    Raises:
      KeyError: If initializers contains any keys other than 'w' or 'b'.
      KeyError: If regularizers contains any keys other than 'w' or 'b'.
      ValueError: If output_sizes is empty.
      TypeError: If `activation` is not callable; or if `output_sizes` is not
        iterable.
    """
    super(MLP, self).__init__(custom_getter=custom_getter, name=name)

    if not isinstance(output_sizes, collections.Iterable):
      raise TypeError("output_sizes must be iterable")
    output_sizes = tuple(output_sizes)
    if not output_sizes:
      raise ValueError("output_sizes must not be empty")
    self._output_sizes = output_sizes
    self._num_layers = len(self._output_sizes)
    self._input_shape = None

    self.possible_keys = self.get_possible_initializer_keys(use_bias=use_bias)
    self._initializers = util.check_initializers(
        initializers, self.possible_keys)
    self._partitioners = util.check_partitioners(
        partitioners, self.possible_keys)
    self._regularizers = util.check_regularizers(
        regularizers, self.possible_keys)
    if not callable(activation):
      raise TypeError("Input 'activation' must be callable")
    self._activation = activation
    self._activate_final = activate_final

    self._use_bias = use_bias
    self._use_dropout = use_dropout
    self._instantiate_layers()

  def _instantiate_layers(self):
    """Instantiates all the linear modules used in the network.

    Layers are instantiated in the constructor, as opposed to the build
    function, because MLP implements the Transposable interface, and the
    transpose function can be called before the module is actually connected
    to the graph and build is called.

    Notice that this is safe since layers in the transposed module are
    instantiated using a lambda returning input_size of the mlp layers, and
    this doesn't have to return sensible values until the original module is
    connected to the graph.
    """

    # Here we are entering the module's variable scope to name our submodules
    # correctly (not to create variables). As such it's safe to not check
    # whether we're in the same graph. This is important if we're constructing
    # the module in one graph and connecting it in another (e.g. with `defun`
    # the module is created in some default graph, and connected to a capturing
    # graph in order to turn it into a graph function).
    with self._enter_variable_scope(check_same_graph=False):
      self._layers = [basic.Linear(self._output_sizes[i],
                                   name="linear_{}".format(i),
                                   initializers=self._initializers,
                                   partitioners=self._partitioners,
                                   regularizers=self._regularizers,
                                   use_bias=self.use_bias)
                      for i in xrange(self._num_layers)]

  @classmethod
  def get_possible_initializer_keys(cls, use_bias=True):
    return basic.Linear.get_possible_initializer_keys(use_bias=use_bias)

  def _build(self, inputs, is_training=True, dropout_keep_prob=0.5):
    """Assembles the `MLP` and connects it to the graph.

    Args:
      inputs: A 2D Tensor of size `[batch_size, input_size]`.
      is_training: A bool or tf.Bool Tensor. Indicates whether we are
        currently training. Defaults to `True`.
      dropout_keep_prob: The probability that each element is kept when
        both `use_dropout` and `is_training` are True. Defaults to 0.5.
    Returns:
      A 2D Tensor of size `[batch_size, output_sizes[-1]]`.
    """
    self._input_shape = tuple(inputs.get_shape().as_list())
    net = inputs

    final_index = self._num_layers - 1
    for layer_id in xrange(self._num_layers):
      net = self._layers[layer_id](net)

      if final_index != layer_id or self._activate_final:
        # Only perform dropout whenever we are activating the layer's outputs.
        if self._use_dropout:
          keep_prob = utils.smart_cond(
              is_training, true_fn=lambda: dropout_keep_prob,
              false_fn=lambda: tf.constant(1.0)
          )
          net = tf.nn.dropout(net, keep_prob=keep_prob)
        net = self._activation(net)

    return net

  @property
  def layers(self):
    """Returns a tuple containing the linear layers of the `MLP`."""
    return self._layers

  @property
  def output_sizes(self):
    """Returns a tuple of all output sizes of all the layers."""
    return tuple([l() if callable(l) else l for l in self._output_sizes])

  @property
  def output_size(self):
    """Returns the size of the module output, not including the batch dimension.

    This allows the MLP to be used inside a DeepRNN.

    Returns:
      The scalar size of the module output.
    """
    last_size = self._output_sizes[-1]
    return last_size() if callable(last_size) else last_size

  @property
  def use_bias(self):
    return self._use_bias

  @property
  def use_dropout(self):
    return self._use_dropout

  @property
  def initializers(self):
    """Returns the intializers dictionary."""
    return self._initializers

  @property
  def partitioners(self):
    """Returns the partitioners dictionary."""
    return self._partitioners

  @property
  def regularizers(self):
    """Returns the regularizers dictionary."""
    return self._regularizers

  @property
  def activation(self):
    return self._activation

  @property
  def activate_final(self):
    return self._activate_final

  # Implements Transposable interface
  @property
  def input_shape(self):
    """Returns shape of input `Tensor` passed at last call to `build`."""
    self._ensure_is_connected()
    return self._input_shape

  # Implements Transposable interface
  def transpose(self, name=None, activate_final=None):
    """Returns transposed `MLP`.

    Args:
      name: Optional string specifying the name of the transposed module. The
        default name is constructed by appending "_transpose"
        to `self.module_name`.
      activate_final: Optional boolean determining if the activation and batch
        normalization, if turned on, are applied to the final layer.

    Returns:
      Matching transposed `MLP` module.
    """
    if name is None:
      name = self.module_name + "_transpose"
    if activate_final is None:
      activate_final = self.activate_final
    output_sizes = [lambda l=layer: l.input_shape[1] for layer in self._layers]
    output_sizes.reverse()
    return MLP(
        name=name,
        output_sizes=output_sizes,
        activation=self.activation,
        activate_final=activate_final,
        initializers=self.initializers,
        partitioners=self.partitioners,
        regularizers=self.regularizers,
        use_bias=self.use_bias,
        use_dropout=self.use_dropout)

  def clone(self, name=None):
    """Creates a new MLP with the same structure.

    Args:
      name: Optional string specifying the name of the new module. The default
        name is constructed by appending "_clone" to the original name.

    Returns:
      A cloned `MLP` module.
    """

    if name is None:
      name = self.module_name + "_clone"
    return MLP(
        name=name,
        output_sizes=self.output_sizes,
        activation=self.activation,
        activate_final=self.activate_final,
        initializers=self.initializers,
        partitioners=self.partitioners,
        regularizers=self.regularizers,
        use_bias=self.use_bias,
        use_dropout=self.use_dropout)
