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

"""Test sonnet.python.modules.nets.convnet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from functools import partial
import itertools
# Dependency imports

import numpy as np
import sonnet as snt
from sonnet.python.modules.conv import _fill_shape as fill_shape
from sonnet.testing import parameterized

import tensorflow as tf
from tensorflow.python.ops import variables


class SharedConvNets2DTest(parameterized.ParameterizedTestCase,
                           tf.test.TestCase):

  def setUp(self):
    super(SharedConvNets2DTest, self).setUp()
    self.output_channels = [2, 3, 4]
    self.kernel_shapes = [[3, 3]]
    self.strides = [1]
    self.paddings = [snt.SAME]

  @parameterized.NamedParameters(
      ("ConvNet2D", snt.nets.ConvNet2D),
      ("ConvNet2DTranspose", partial(snt.nets.ConvNet2DTranspose,
                                     output_shapes=[[100, 100]])))
  def testName(self, module):
    unique_name = "unique_name"
    with tf.variable_scope("scope"):
      net = module(name=unique_name,
                   output_channels=self.output_channels,
                   kernel_shapes=self.kernel_shapes,
                   strides=self.strides,
                   paddings=self.paddings)
    self.assertEqual(net.scope_name, "scope/" + unique_name)
    self.assertEqual(net.module_name, unique_name)

  @parameterized.NamedParameters(
      ("ConvNet2D", snt.nets.ConvNet2D),
      ("ConvNet2DTranspose", partial(snt.nets.ConvNet2DTranspose,
                                     output_shapes=[[100, 100]])))
  def testConstructor(self, module):
    with self.assertRaisesRegexp(ValueError,
                                 "output_channels must not be empty"):
      module(output_channels=[],
             kernel_shapes=self.kernel_shapes,
             strides=self.strides,
             paddings=self.paddings)

    with self.assertRaisesRegexp(ValueError,
                                 "kernel_shapes must be of length 1 or *"):
      module(output_channels=self.output_channels,
             kernel_shapes=[],
             strides=self.strides,
             paddings=self.paddings)

    with self.assertRaisesRegexp(ValueError,
                                 "kernel_shapes must be of length 1 or *"):
      module(output_channels=self.output_channels,
             kernel_shapes=[1, 2],
             strides=self.strides,
             paddings=self.paddings)

    with self.assertRaisesRegexp(ValueError,
                                 "strides must be of length 1 or *"):
      module(output_channels=self.output_channels,
             kernel_shapes=self.kernel_shapes,
             strides=[],
             paddings=self.paddings)

    with self.assertRaisesRegexp(ValueError,
                                 "strides must be of length 1 or *"):
      module(output_channels=self.output_channels,
             kernel_shapes=self.kernel_shapes,
             strides=[1, 1],
             paddings=self.paddings)

    with self.assertRaisesRegexp(ValueError,
                                 "paddings must be of length 1 or *"):
      module(output_channels=self.output_channels,
             kernel_shapes=self.kernel_shapes,
             strides=self.paddings,
             paddings=[])

    with self.assertRaisesRegexp(ValueError,
                                 "paddings must be of length 1 or *"):
      module(output_channels=self.output_channels,
             kernel_shapes=self.kernel_shapes,
             strides=self.strides,
             paddings=[snt.SAME, snt.SAME])

    with self.assertRaisesRegexp(KeyError,
                                 "Invalid initializer keys.*"):
      module(
          output_channels=self.output_channels,
          kernel_shapes=self.kernel_shapes,
          strides=self.strides,
          paddings=self.paddings,
          initializers={"not_w": tf.truncated_normal_initializer(stddev=1.0)})

    with self.assertRaisesRegexp(TypeError,
                                 "Initializer for 'w' is not a callable "
                                 "function or dictionary"):
      module(output_channels=self.output_channels,
             kernel_shapes=self.kernel_shapes,
             strides=self.strides,
             paddings=self.paddings,
             initializers={"w": tf.zeros([1, 2, 3])})

    with self.assertRaisesRegexp(KeyError,
                                 "Invalid regularizer keys.*"):
      module(
          output_channels=self.output_channels,
          kernel_shapes=self.kernel_shapes,
          strides=self.strides,
          paddings=self.paddings,
          regularizers={"not_w": tf.contrib.layers.l1_regularizer(scale=0.5)})

    with self.assertRaisesRegexp(TypeError,
                                 "Regularizer for 'w' is not a callable "
                                 "function or dictionary"):
      module(output_channels=self.output_channels,
             kernel_shapes=self.kernel_shapes,
             strides=self.strides,
             paddings=self.paddings,
             regularizers={"w": tf.zeros([1, 2, 3])})

    with self.assertRaisesRegexp(TypeError,
                                 "Input 'activation' must be callable"):
      module(output_channels=self.output_channels,
             kernel_shapes=self.kernel_shapes,
             strides=self.strides,
             paddings=self.paddings,
             activation="not_a_function")

    err = "`batch_norm_config` must be a mapping, e.g. `dict`"
    with self.assertRaisesRegexp(TypeError, err):
      module(output_channels=self.output_channels,
             kernel_shapes=self.kernel_shapes,
             strides=self.strides,
             paddings=self.paddings,
             batch_norm_config="not a valid config")

    err = "output_channels must be iterable"
    with self.assertRaisesRegexp(TypeError, err):
      module(output_channels=42,
             kernel_shapes=self.kernel_shapes,
             strides=self.strides,
             paddings=self.paddings)

    err = "kernel_shapes must be iterable"
    with self.assertRaisesRegexp(TypeError, err):
      module(output_channels=self.output_channels,
             kernel_shapes=None,
             strides=self.strides,
             paddings=self.paddings)

    err = "strides must be iterable"
    with self.assertRaisesRegexp(TypeError, err):
      module(output_channels=self.output_channels,
             kernel_shapes=self.kernel_shapes,
             strides=True,
             paddings=self.paddings)

    err = "paddings must be iterable"
    with self.assertRaisesRegexp(TypeError, err):
      module(output_channels=self.output_channels,
             kernel_shapes=self.kernel_shapes,
             strides=self.strides,
             paddings=lambda x: x + 42)

    err = "use_bias must be either a bool or an iterable"
    with self.assertRaisesRegexp(TypeError, err):
      module(output_channels=self.output_channels,
             kernel_shapes=self.kernel_shapes,
             strides=self.strides,
             paddings=self.paddings,
             use_bias=2)

  @parameterized.NamedParameters(
      ("ConvNet2D", snt.nets.ConvNet2D),
      ("ConvNet2DTranspose",
       partial(snt.nets.ConvNet2DTranspose,
               output_shapes=[[100, 100]])))
  def testBatchNormBuildFlag(self, module):
    model = module(output_channels=self.output_channels,
                   kernel_shapes=self.kernel_shapes,
                   strides=self.strides,
                   paddings=self.paddings,
                   use_batch_norm=True)
    self.assertTrue(model.use_batch_norm)
    input_to_net = tf.placeholder(tf.float32, shape=(1, 100, 100, 3))

    # Check that an error is raised if we don't specify the is_training flag
    err = "is_training flag must be explicitly specified"
    with self.assertRaisesRegexp(ValueError, err):
      model(input_to_net)

  @parameterized.NamedParameters(
      ("ConvNet2D", snt.nets.ConvNet2D),
      ("ConvNet2DTranspose",
       partial(snt.nets.ConvNet2DTranspose,
               output_shapes=[[100, 100]])))
  def testBatchNorm(self, module):
    model = module(output_channels=self.output_channels,
                   kernel_shapes=self.kernel_shapes,
                   strides=self.strides,
                   paddings=self.paddings,
                   use_batch_norm=True)
    self.assertTrue(model.use_batch_norm)
    input_to_net = tf.placeholder(tf.float32, shape=(1, 100, 100, 3))

    # Check Tensorflow flags work
    is_training = tf.placeholder(tf.bool)
    test_local_stats = tf.placeholder(tf.bool)

    model(input_to_net,
          is_training=is_training,
          test_local_stats=test_local_stats)

    # Check Python is_training flag works
    model(input_to_net, is_training=False, test_local_stats=False)

    model_variables = model.get_variables()

    self.assertEqual(
        len(model_variables),
        len(self.output_channels) * 3 - 1)

    # Check that the appropriate moving statistics variables have been created.
    self.assertTrue(
        any("moving_variance" in var.name
            for var in tf.global_variables()))
    self.assertTrue(
        any("moving_mean" in var.name
            for var in tf.global_variables()))

  @parameterized.NamedParameters(
      ("ConvNet2D", snt.nets.ConvNet2D),
      ("ConvNet2DTranspose", partial(snt.nets.ConvNet2DTranspose,
                                     output_shapes=[[100, 100]])))
  def testBatchNormConfig(self, module):
    batch_norm_config = {
        "scale": True,
    }

    model = module(output_channels=self.output_channels,
                   kernel_shapes=self.kernel_shapes,
                   strides=self.strides,
                   paddings=self.paddings,
                   use_batch_norm=True,
                   batch_norm_config=batch_norm_config)

    input_to_net = tf.placeholder(tf.float32, shape=(1, 100, 100, 3))

    model(input_to_net, is_training=True)
    model_variables = model.get_variables()

    self.assertEqual(
        len(model_variables),
        len(self.output_channels) * 4 - 2)

  @parameterized.NamedParameters(
      ("ConvNet2D", snt.nets.ConvNet2D),
      ("ConvNet2DTranspose", partial(snt.nets.ConvNet2DTranspose,
                                     output_shapes=[[100, 100]])))
  def testNoBias(self, module):
    model = module(output_channels=self.output_channels,
                   kernel_shapes=self.kernel_shapes,
                   strides=self.strides,
                   paddings=self.paddings,
                   use_bias=False)
    self.assertEqual(model.use_bias, (False,) * len(self.output_channels))
    input_to_net = tf.placeholder(tf.float32, shape=(1, 100, 100, 3))
    model(input_to_net)

    model_variables = model.get_variables()

    self.assertEqual(
        len(model_variables),
        len(self.output_channels))

  @parameterized.NamedParameters(
      ("ConvNet2D", snt.nets.ConvNet2D),
      ("ConvNet2DTranspose", partial(snt.nets.ConvNet2DTranspose,
                                     output_shapes=[[100, 100]])))
  def testNoBiasIterable(self, module):
    use_bias = (True,) * (len(self.output_channels) - 1) + (False,)
    model = module(output_channels=self.output_channels,
                   kernel_shapes=self.kernel_shapes,
                   strides=self.strides,
                   paddings=self.paddings,
                   use_bias=use_bias)

    actual_use_biases = tuple(layer.has_bias for layer in model.layers)
    self.assertEqual(model.use_bias, actual_use_biases)
    self.assertEqual(use_bias, actual_use_biases)

    model_transpose = model.transpose()
    actual_use_biases = tuple(layer.has_bias
                              for layer in model_transpose.layers)
    self.assertEqual(model_transpose.use_bias, actual_use_biases)
    self.assertEqual(tuple(reversed(use_bias)), actual_use_biases)

  @parameterized.NamedParameters(
      ("ConvNet2DNoBias", snt.nets.ConvNet2D, False),
      ("ConvNet2DBias", snt.nets.ConvNet2D, True),
      ("ConvNet2DTransposeNoBias", partial(snt.nets.ConvNet2DTranspose,
                                           output_shapes=[[100, 100]]), False),
      ("ConvNet2DTransposeBias", partial(snt.nets.ConvNet2DTranspose,
                                         output_shapes=[[100, 100]]), True))
  def testRegularizersInRegularizationLosses(self, module, use_bias):
    if use_bias:
      regularizers = {"w": tf.contrib.layers.l1_regularizer(scale=0.5),
                      "b": tf.contrib.layers.l2_regularizer(scale=0.5)}
    else:
      regularizers = {"w": tf.contrib.layers.l1_regularizer(scale=0.5)}

    model = module(output_channels=self.output_channels,
                   kernel_shapes=self.kernel_shapes,
                   strides=self.strides,
                   paddings=self.paddings,
                   use_bias=use_bias,
                   regularizers=regularizers)

    input_to_net = tf.placeholder(tf.float32, shape=(1, 100, 100, 3))
    model(input_to_net)

    graph_regularizers = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    self.assertRegexpMatches(graph_regularizers[0].name, ".*l1_regularizer.*")
    if use_bias:
      self.assertRegexpMatches(graph_regularizers[1].name, ".*l2_regularizer.*")

  @parameterized.NamedParameters(
      ("ConvNet2D", snt.nets.ConvNet2D, False),
      ("ConvNet2DFinal", snt.nets.ConvNet2D, True),
      ("ConvNet2DTranspose",
       partial(snt.nets.ConvNet2DTranspose, output_shapes=[[100, 100]]),
       False),
      ("ConvNet2DTransposeFinal",
       partial(snt.nets.ConvNet2DTranspose, output_shapes=[[100, 100]]),
       True))
  def testActivateFinal(self, module, activate_final):
    model = module(output_channels=self.output_channels,
                   kernel_shapes=self.kernel_shapes,
                   strides=self.strides,
                   paddings=self.paddings,
                   activate_final=activate_final,
                   use_batch_norm=True,
                   use_bias=False)
    self.assertEqual(activate_final, model.activate_final)
    input_to_net = tf.placeholder(tf.float32, shape=(1, 100, 100, 3))
    model(input_to_net, is_training=True)

    model_variables = model.get_variables()

    # Batch norm variable missing for final activation
    if activate_final:
      self.assertEqual(len(model_variables), len(self.output_channels) * 2)
    else:
      self.assertEqual(len(model_variables), len(self.output_channels) * 2 - 1)

    # Test transpose method's activate_final arg.
    transposed_model_activate_final = model.transpose(activate_final=True)
    transposed_model_no_activate_final = model.transpose(activate_final=False)
    transposed_model_inherit_activate_final = model.transpose()
    self.assertEqual(True, transposed_model_activate_final.activate_final)
    self.assertEqual(False, transposed_model_no_activate_final.activate_final)
    self.assertEqual(model.activate_final,
                     transposed_model_inherit_activate_final.activate_final)

  @parameterized.Parameters(
      *itertools.product(
          [snt.nets.ConvNet2D,
           partial(snt.nets.ConvNet2DTranspose, output_shapes=[[100, 100]])],
          ["kernel_shapes", "strides", "paddings", "activation", "initializers",
           "partitioners", "regularizers", "use_bias", "batch_norm_config"]))
  def testTransposeDefaultParameter(self, module, param_name):
    """Tests if .transpose correctly chooses the default parameters.

    Args:
      module: The conv net class.
      param_name: The name of the parameter to test.
    """
    # For these parameters, the expected values are their reversed values
    expected_reversed = ["kernel_shapes", "strides", "paddings", "use_bias"]

    # We have to choose asymmetric parameter values here in order for the test
    # to be effective. This is why we don't take the default ones.
    model = module(output_channels=[2, 3, 4],
                   kernel_shapes=[[3, 3], [5, 5], [7, 7]],
                   strides=[[1, 1], [2, 2], [3, 3]],
                   paddings=[snt.SAME, snt.SAME, snt.VALID],
                   use_batch_norm=[True, True, False],
                   use_bias=[True, True, False])

    # We don't pass the parameter on to .transpose, None should be the default
    transpose_model = model.transpose()
    if param_name in expected_reversed:
      self.assertItemsEqual(reversed(getattr(model, param_name)),
                            getattr(transpose_model, param_name))
    else:
      self.assertEqual(getattr(model, param_name),
                       getattr(transpose_model, param_name))

  @parameterized.Parameters(
      *itertools.product(
          [snt.nets.ConvNet2D,
           partial(snt.nets.ConvNet2DTranspose, output_shapes=[[100, 100]])],
          [("kernel_shapes", [[3, 3], [3, 3], [3, 3]]),
           ("strides", [[1, 1], [1, 1], [1, 1]]),
           ("paddings", [snt.SAME, snt.SAME, snt.SAME]),
           ("activation", tf.nn.tanh),
           ("initializers", {}),
           ("partitioners", {}),
           ("regularizers", {}),
           ("use_bias", [True, True, True]),
           ("batch_norm_config", {"scale": True})]))
  def testTransposePassThroughParameter(self, module, param_name_and_value):
    """Tests if .transpose correctly passes through the given parameters.

    Args:
      module: The conv net class.
      param_name_and_value: Tuple consisting of the parameter name and value.
    """
    param_name, param_value = param_name_and_value
    # The given parameter values are all for three-layer networks. Changing
    # the default parameters would therefore break this test. Thus, we choose
    # fixed/independent parameters.
    model = module(output_channels=[2, 3, 4],
                   kernel_shapes=[[3, 3], [5, 5], [7, 7]],
                   strides=[[1, 1], [2, 2], [3, 3]],
                   paddings=[snt.SAME, snt.SAME, snt.VALID],
                   use_batch_norm=[True, True, False],
                   use_bias=[True, True, False])

    transpose_model = model.transpose(**{param_name: param_value})
    if isinstance(param_value, collections.Iterable):
      self.assertItemsEqual(param_value, getattr(transpose_model, param_name))
    else:
      self.assertEqual(param_value, getattr(transpose_model, param_name))


class ConvNet2DTest(parameterized.ParameterizedTestCase,
                    tf.test.TestCase):

  def setUp(self):
    super(ConvNet2DTest, self).setUp()
    self.output_channels = [2, 3, 4]
    self.kernel_shapes = [[3, 3]]
    self.strides = [1]
    self.paddings = [snt.SAME]

  def testConstructor(self):
    net = snt.nets.ConvNet2D(output_channels=self.output_channels,
                             kernel_shapes=self.kernel_shapes,
                             strides=self.strides,
                             paddings=self.paddings)
    self.assertEqual(len(net.layers), len(self.output_channels))

    for i, layer in enumerate(net.layers):
      self.assertEqual(layer.output_channels, self.output_channels[i])
      self.assertEqual(layer.stride,
                       (1,) + fill_shape(self.strides[0], 2) + (1,))
      self.assertEqual(layer.kernel_shape, fill_shape(self.kernel_shapes[0], 2))
      self.assertEqual(layer.padding, self.paddings[0])
      self.assertEqual(layer.output_channels, net.output_channels[i])
      self.assertEqual(layer.stride,
                       (1,) + fill_shape(net.strides[i], 2) + (1,))
      self.assertEqual(layer.kernel_shape, fill_shape(net.kernel_shapes[i], 2))
      self.assertEqual(layer.padding, net.paddings[i])

  def testTranspose(self):
    with tf.variable_scope("scope1"):
      net = snt.nets.ConvNet2D(output_channels=self.output_channels,
                               kernel_shapes=self.kernel_shapes,
                               strides=self.strides,
                               paddings=self.paddings,
                               name="conv_net_2d")

    err = "Iterable output_channels length must match the number of layers"
    with self.assertRaisesRegexp(ValueError, err):
      net.transpose(output_channels=[42] * 18)

    with tf.variable_scope("scope2"):
      net_transpose = net.transpose()

    self.assertEqual("scope1/conv_net_2d", net.scope_name)
    self.assertEqual("conv_net_2d", net.module_name)
    self.assertEqual("scope2/conv_net_2d_transpose", net_transpose.scope_name)
    self.assertEqual("conv_net_2d_transpose", net_transpose.module_name)

    input_shape = [10, 100, 100, 3]
    input_to_net = tf.placeholder(tf.float32, shape=input_shape)
    # Tests that trying to connect the trasposed network before connecting the
    # original nets raises an error. The reason is that the output_shapes and
    # output_channels are laziliy evaluated and not yet known.
    with self.assertRaisesRegexp(snt.Error,
                                 "Variables in {} not instantiated yet, "
                                 "__call__ the module first.".format(
                                     net.layers[-1].scope_name)):
      net_transpose(input_to_net)

    net_transpose = net.transpose(name="another_net_transpose")
    net_out = net(input_to_net, is_training=True)
    self.assertEqual(net.input_shape, tuple(input_shape))
    net_transposed_output = net_transpose(net_out)
    self.assertEqual(net_transposed_output.get_shape(),
                     input_to_net.get_shape())
    for i in range(len(net.layers)):
      self.assertEqual(net_transpose.layers[i].output_shape,
                       net.layers[-1 - i].input_shape[1:-1])
      self.assertEqual(net_transpose.layers[i].output_channels,
                       net.layers[-1 - i].input_shape[-1])

    data = np.random.rand(*input_shape)
    init = tf.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(init)
      sess.run(net_transposed_output, feed_dict={input_to_net: data})

  def testVariableMap(self):
    """Tests for regressions in variable names."""

    use_bias = True
    use_batch_norm = True
    var_names_w = [
        u"conv_net_2d/conv_2d_0/w:0",
        u"conv_net_2d/conv_2d_1/w:0",
        u"conv_net_2d/conv_2d_2/w:0",
    ]
    var_names_b = [
        u"conv_net_2d/conv_2d_0/b:0",
        u"conv_net_2d/conv_2d_1/b:0",
        u"conv_net_2d/conv_2d_2/b:0",
    ]
    var_names_bn = [
        u"conv_net_2d/batch_norm_0/beta:0",
        u"conv_net_2d/batch_norm_1/beta:0",
    ]

    correct_variable_names = set(var_names_w + var_names_b + var_names_bn)

    module = snt.nets.ConvNet2D(output_channels=self.output_channels,
                                kernel_shapes=self.kernel_shapes,
                                strides=self.strides,
                                paddings=self.paddings,
                                use_bias=use_bias,
                                use_batch_norm=use_batch_norm)

    input_shape = [10, 100, 100, 3]
    input_to_net = tf.placeholder(tf.float32, shape=input_shape)

    _ = module(input_to_net, is_training=True)

    variable_names = [var.name for var in module.get_variables()]

    self.assertEqual(set(variable_names), correct_variable_names)

  def testPartitioners(self):
    partitioners = {
        "w": tf.variable_axis_size_partitioner(10),
        "b": tf.variable_axis_size_partitioner(8),
    }

    module = snt.nets.ConvNet2D(output_channels=self.output_channels,
                                kernel_shapes=self.kernel_shapes,
                                strides=self.strides,
                                paddings=self.paddings,
                                partitioners=partitioners)

    input_shape = [10, 100, 100, 3]
    input_to_net = tf.placeholder(tf.float32, shape=input_shape)

    _ = module(input_to_net)

    for layer in module._layers:
      self.assertEqual(type(layer.w), variables.PartitionedVariable)
      self.assertEqual(type(layer.b), variables.PartitionedVariable)


class ConvNet2DTransposeTest(tf.test.TestCase):

  def setUp(self):
    super(ConvNet2DTransposeTest, self).setUp()
    self.output_channels = [2, 3, 4]
    self.output_shapes = [[100, 100]]
    self.kernel_shapes = [[3, 3]]
    self.strides = [1]
    self.paddings = [snt.SAME]

  def testConstructor(self):

    with self.assertRaisesRegexp(ValueError,
                                 "output_shapes must be of length 1 or *"):
      snt.nets.ConvNet2DTranspose(output_channels=self.output_channels,
                                  output_shapes=[],
                                  kernel_shapes=self.kernel_shapes,
                                  strides=self.strides,
                                  paddings=self.paddings)

    with self.assertRaisesRegexp(ValueError,
                                 "output_shapes must be of length 1 or *"):
      snt.nets.ConvNet2DTranspose(output_channels=self.output_channels,
                                  output_shapes=[[1, 2], [1, 2]],
                                  kernel_shapes=self.kernel_shapes,
                                  strides=[],
                                  paddings=self.paddings)

    with self.assertRaisesRegexp(KeyError,
                                 "Invalid initializer keys.*"):
      snt.nets.ConvNet2DTranspose(
          output_channels=self.output_channels,
          output_shapes=self.output_shapes,
          kernel_shapes=self.kernel_shapes,
          strides=self.strides,
          paddings=self.paddings,
          initializers={"not_w": tf.truncated_normal_initializer(stddev=1.0)})

    net = snt.nets.ConvNet2DTranspose(output_channels=self.output_channels,
                                      output_shapes=self.output_shapes,
                                      kernel_shapes=self.kernel_shapes,
                                      strides=self.strides,
                                      paddings=self.paddings)
    self.assertEqual(net.output_shapes,
                     tuple(self.output_shapes) * len(self.output_channels))
    self.assertEqual(len(net.layers), len(self.output_channels))

    for i, layer in enumerate(net.layers):
      self.assertEqual(layer.output_channels, self.output_channels[i])
      self.assertEqual(layer.stride,
                       (1,) + fill_shape(self.strides[0], 2) + (1,))
      self.assertEqual(layer.kernel_shape, fill_shape(self.kernel_shapes[0], 2))
      self.assertEqual(layer.padding, self.paddings[0])
      self.assertEqual(layer.output_channels, net.output_channels[i])
      self.assertEqual(layer.stride,
                       (1,) + fill_shape(net.strides[i], 2) + (1,))
      self.assertEqual(layer.kernel_shape, fill_shape(net.kernel_shapes[i], 2))
      self.assertEqual(layer.padding, net.paddings[i])

    with self.assertRaisesRegexp(TypeError, "output_shapes must be iterable"):
      snt.nets.ConvNet2DTranspose(output_channels=self.output_channels,
                                  output_shapes=False,
                                  kernel_shapes=self.kernel_shapes,
                                  strides=self.strides,
                                  paddings=self.paddings)

  def testTranspose(self):
    net = snt.nets.ConvNet2DTranspose(output_channels=self.output_channels,
                                      output_shapes=self.output_shapes,
                                      kernel_shapes=self.kernel_shapes,
                                      strides=self.strides,
                                      paddings=self.paddings)

    err = "Iterable output_channels length must match the number of layers"
    with self.assertRaisesRegexp(ValueError, err):
      net.transpose(output_channels=[42] * 18)
    net_transpose = net.transpose()
    input_shape = [10, 100, 100, 3]
    input_to_net = tf.placeholder(tf.float32, shape=input_shape)
    # Tests that trying to connect the trasposed network before connecting the
    # original nets raises an error. The reason is that the output_shapes and
    # output_channels are laziliy evaluated and not yet known.
    with self.assertRaisesRegexp(snt.Error,
                                 "Variables in {} not instantiated yet, "
                                 "__call__ the module first.".format(
                                     net.layers[-1].scope_name)):
      net_transpose(input_to_net)

    net_transpose = net.transpose(name="another_net_transpose")
    net_out = net(input_to_net, is_training=True)
    net_transposed_output = net_transpose(net_out)
    self.assertEqual(net_transposed_output.get_shape(),
                     input_to_net.get_shape())
    for i in range(len(net.layers)):
      self.assertEqual(net_transpose.layers[i].input_shape[1:-1],
                       net.layers[-1 - i].output_shape)
      self.assertEqual(net_transpose.layers[i].output_channels,
                       net.layers[-1 - i].input_shape[-1])

    data = np.random.rand(*input_shape)
    init = tf.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(init)
      sess.run(net_transposed_output, feed_dict={input_to_net: data})

  def testPartitioners(self):
    partitioners = {
        "w": tf.variable_axis_size_partitioner(10),
        "b": tf.variable_axis_size_partitioner(8),
    }

    module = snt.nets.ConvNet2DTranspose(output_channels=self.output_channels,
                                         output_shapes=self.output_shapes,
                                         kernel_shapes=self.kernel_shapes,
                                         strides=self.strides,
                                         paddings=self.paddings,
                                         partitioners=partitioners)

    input_shape = [10, 100, 100, 3]
    input_to_net = tf.placeholder(tf.float32, shape=input_shape)

    _ = module(input_to_net)

    for layer in module._layers:
      self.assertEqual(type(layer.w), variables.PartitionedVariable)
      self.assertEqual(type(layer.b), variables.PartitionedVariable)


if __name__ == "__main__":
  tf.test.main()
