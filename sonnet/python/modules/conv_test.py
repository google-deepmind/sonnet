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

"""Tests for `sonnet.python.modules.conv`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import random

# Dependency imports
import numpy as np
import sonnet as snt
from sonnet.python.modules import conv
from sonnet.testing import parameterized
import tensorflow as tf

from tensorflow.python.ops import variables


def create_constant_initializers(w, b, use_bias):
  if use_bias:
    return {"w": tf.constant_initializer(w),
            "b": tf.constant_initializer(b)}
  else:
    return {"w": tf.constant_initializer(w)}


def create_separable_constant_initializers(w_dw, w_pw, b, use_bias):
  if use_bias:
    return {"w_dw": tf.constant_initializer(w_dw),
            "w_pw": tf.constant_initializer(w_pw),
            "b": tf.constant_initializer(b)}
  else:
    return {"w_dw": tf.constant_initializer(w_dw),
            "w_pw": tf.constant_initializer(w_pw)}


def create_regularizers(use_bias, regularizer):
  if use_bias:
    return {"w": regularizer, "b": regularizer}
  else:
    return {"w": regularizer}


def create_separable_regularizers(use_bias, regularizer):
  if use_bias:
    return {"w_dw": regularizer, "w_pw": regularizer, "b": regularizer}
  else:
    return {"w_dw": regularizer, "w_pw": regularizer}


class FillListTest(tf.test.TestCase):

  def test(self):
    """Tests the _fill_list private function in snt.conv."""
    x = random.randint(1, 10)

    self.assertEqual(conv._fill_shape(x, 1), (x,))
    self.assertEqual(conv._fill_shape(x, 2), (x, x))
    self.assertEqual(conv._fill_shape(x, 3), (x, x, x))
    self.assertEqual(conv._fill_shape(x, 4), (x, x, x, x))
    self.assertEqual(conv._fill_shape([x, x + 1, x + 2], 3),
                     (x, x + 1, x + 2))

    err = "n must be a positive integer"
    with self.assertRaisesRegexp(TypeError, err):
      conv._fill_shape(x, 0)

    err = ("must be either a positive integer or an iterable of positive "
           "integers of size 4")
    with self.assertRaisesRegexp(TypeError, err):
      conv._fill_shape([], 4)
    with self.assertRaisesRegexp(TypeError, err):
      conv._fill_shape([x], 4)
    with self.assertRaisesRegexp(TypeError, err):
      conv._fill_shape([x, x], 4)
    with self.assertRaisesRegexp(TypeError, err):
      conv._fill_shape(["b"], 4)


class DefaultTransposeSizeTest(parameterized.ParameterizedTestCase,
                               tf.test.TestCase):

  # Constants for use in parameterized test.
  input_shape = [[20], [23, 11, 13], [1, 3]]
  stride = [[3], [7, 1, 2], [6, 2]]
  kernel_shape = [[4], [1, 3, 2], [34, 2]]
  padding = [snt.SAME, snt.VALID, snt.VALID]
  output_shape = []

  for i, pad in enumerate(padding):
    if pad == snt.SAME:
      output_shape.append([x * y for x, y in zip(input_shape[i], stride[i])])
    if pad == snt.VALID:
      output_shape.append([x * y + z - 1 for x, y, z in
                           zip(input_shape[i], stride[i], kernel_shape[i])])

  @parameterized.Parameters(
      *zip(input_shape, stride, kernel_shape, padding, output_shape))
  def testFunction(self, input_shape, stride, kernel_shape, padding,
                   output_shape):
    """Test output shapes are correct."""
    self.assertEqual(conv._default_transpose_size(input_shape, stride,
                                                  kernel_shape=kernel_shape,
                                                  padding=padding),
                     tuple(output_shape))

  @parameterized.Parameters(
      *zip(input_shape, stride, kernel_shape, padding, output_shape))
  def testModules(self, input_shape, stride, kernel_shape, padding,
                  output_shape):
    """Test ConvTranspose modules return expected default output shapes."""
    if len(input_shape) == 1:
      module = snt.Conv1DTranspose
    elif len(input_shape) == 2:
      module = snt.Conv2DTranspose
    elif len(input_shape) == 3:
      module = snt.Conv3DTranspose

    batch_size = [1]
    channels = [1]

    inputs = tf.zeros(shape=batch_size + input_shape + channels,
                      dtype=tf.float32)
    outputs = module(output_channels=1, kernel_shape=kernel_shape,
                     stride=stride, padding=padding)(inputs)
    self.assertEqual(output_shape, outputs.get_shape().as_list()[1:-1])

  @parameterized.Parameters(
      *zip(input_shape, stride, kernel_shape, padding, output_shape))
  def testConnectTwice(self, input_shape, stride, kernel_shape, padding,
                       output_shape):
    """Test ConvTranspose modules with multiple connections."""
    if len(input_shape) == 1:
      module = snt.Conv1DTranspose
    elif len(input_shape) == 2:
      module = snt.Conv2DTranspose
    elif len(input_shape) == 3:
      module = snt.Conv3DTranspose

    batch_size = [1]
    channels = [1]

    inputs = tf.zeros(shape=batch_size + input_shape + channels,
                      dtype=tf.float32)
    inputs_2 = tf.zeros(shape=batch_size + input_shape + channels,
                        dtype=tf.float32)
    conv1 = module(output_channels=1, kernel_shape=kernel_shape,
                   stride=stride, padding=padding)
    outputs = conv1(inputs)

    # Connecting for the second time with the same shape should be OK.
    outputs_2 = conv1(inputs_2)

    # So should connecting with a different shape.
    new_input_shape = [25] * len(input_shape)
    new_inputs = tf.zeros(shape=batch_size + new_input_shape + channels,
                          dtype=tf.float32)
    new_outputs = conv1(new_inputs)

    with self.test_session() as sess:
      tf.global_variables_initializer().run()
      outputs_array, outputs_array_2 = sess.run([outputs, outputs_2])
      self.assertEqual(outputs_array.shape, outputs_array_2.shape)

      sess.run(new_outputs)


class SharedConvTest(parameterized.ParameterizedTestCase, tf.test.TestCase):

  CONV_1D_KWARGS = {
      "output_channels": 1,
      "kernel_shape": 3,
  }
  CONV_2D_KWARGS = CONV_1D_KWARGS
  CONV_3D_KWARGS = CONV_1D_KWARGS
  DEPTHWISE_CONV_2D_KWARGS = {
      "channel_multiplier": 1,
      "kernel_shape": 3,
  }
  SEPARABLE_CONV_2D_KWARGS = {
      "output_channels": 10,
      "channel_multiplier": 1,
      "kernel_shape": 3,
  }
  IN_PLANE_CONV_2D_KWARGS = {
      "kernel_shape": 3,
  }
  CONV_1D_TRANSPOSE_KWARGS = {
      "output_channels": 1,
      "output_shape": [10],
      "kernel_shape": 3,
  }
  CONV_2D_TRANSPOSE_KWARGS = {
      "output_channels": 1,
      "output_shape": [10, 10],
      "kernel_shape": 3,
  }
  CONV_3D_TRANSPOSE_KWARGS = {
      "output_channels": 1,
      "output_shape": [10, 10, 10],
      "kernel_shape": 3,
  }

  modules = [
      (snt.Conv1D, 1, CONV_1D_KWARGS),
      (snt.Conv2D, 2, CONV_2D_KWARGS),
      (snt.Conv3D, 3, CONV_3D_KWARGS),
      (snt.Conv1DTranspose, 1, CONV_1D_TRANSPOSE_KWARGS),
      (snt.Conv2DTranspose, 2, CONV_2D_TRANSPOSE_KWARGS),
      (snt.Conv3DTranspose, 3, CONV_3D_TRANSPOSE_KWARGS),
      (snt.DepthwiseConv2D, 2, DEPTHWISE_CONV_2D_KWARGS),
      (snt.InPlaneConv2D, 2, IN_PLANE_CONV_2D_KWARGS),
      (snt.SeparableConv2D, 2, SEPARABLE_CONV_2D_KWARGS),
  ]

  @parameterized.Parameters(*modules)
  def testPartitioners(self, module, num_input_dims, module_kwargs):
    inputs = tf.zeros((10,) * (num_input_dims + 2))

    keys = module.get_possible_initializer_keys(use_bias=True)
    partitioners = {
        key: tf.variable_axis_size_partitioner(10) for key in keys
    }
    convolution = module(partitioners=partitioners, **module_kwargs)
    convolution(inputs)

    for key in keys:
      self.assertEqual(type(getattr(convolution, key)),
                       variables.PartitionedVariable)

    if isinstance(conv, snt.Transposable):
      convolution_t = convolution.transpose()
      self.assertEqual(convolution_t.partitioners, convolution.partitioners)

  @parameterized.Parameters(*itertools.product(modules, (True, False)))
  def testVariables(self, module_info, use_bias):
    """The correct number of variables are created."""
    module, num_input_dims, module_kwargs = module_info

    mod_name = "module"

    input_shape = (10,) * (num_input_dims + 2)
    inputs = tf.placeholder(tf.float32, input_shape)

    with tf.variable_scope("scope"):
      conv_mod = module(name=mod_name, use_bias=use_bias, **module_kwargs)

    self.assertEqual(conv_mod.scope_name, "scope/" + mod_name)
    self.assertEqual(conv_mod.module_name, mod_name)

    with self.assertRaisesRegexp(snt.NotConnectedError, "not instantiated yet"):
      conv_mod.get_variables()

    output = conv_mod(inputs)

    # Check that the graph and module has the correct number of variables: one
    # two, or three, depending on module and configuration.
    supposed_variables = conv_mod.get_possible_initializer_keys(
        use_bias=use_bias)
    self.assertIn(len(supposed_variables), [1, 2, 3])

    graph_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    self.assertEqual(len(graph_variables), len(supposed_variables))
    conv_variables = conv_mod.get_variables()
    self.assertEqual(len(conv_variables), len(supposed_variables))

    variable_names = {v.name for v in conv_variables}

    for var_name in supposed_variables:
      self.assertIn("scope/{}/{}:0".format(mod_name, var_name), variable_names)

    with self.test_session() as sess:
      tf.global_variables_initializer().run()
      inputs_data = np.random.rand(*input_shape)
      sess.run(output, feed_dict={inputs: inputs_data})

  @parameterized.Parameters(*itertools.product(modules, (True, False)))
  def testMissingChannelsError(self, module_info, use_bias):
    """Error is thrown if the input is missing a channel dimension."""
    module, num_input_dims, module_kwargs = module_info
    conv_mod = module(use_bias=use_bias, **module_kwargs)

    inputs = tf.placeholder(tf.float32, (10,) * (num_input_dims + 1))

    err = "Input Tensor must have shape"
    with self.assertRaisesRegexp(snt.IncompatibleShapeError, err):
      conv_mod(inputs)

  @parameterized.Parameters(*itertools.product(modules, (True, False)))
  def testFlattenedError(self, module_info, use_bias):
    """Error is thrown if the input has been incorrectly flattened."""
    module, num_input_dims, module_kwargs = module_info
    conv_mod = module(use_bias=use_bias, **module_kwargs)

    inputs = tf.placeholder(tf.float32, (10,) * (num_input_dims + 1))
    inputs = snt.BatchFlatten()(inputs)

    err = "Input Tensor must have shape"
    with self.assertRaisesRegexp(snt.IncompatibleShapeError, err):
      conv_mod(inputs)

  @parameterized.Parameters(*modules)
  def testCustomGetter(self, module, num_input_dims, module_kwargs):
    """Check that custom_getter option works."""

    def stop_gradient(getter, *args, **kwargs):
      return tf.stop_gradient(getter(*args, **kwargs))

    inputs = tf.placeholder(tf.float32, (10,) * (num_input_dims + 2))

    conv_mod1 = module(**module_kwargs)
    out1 = conv_mod1(inputs)

    conv_mod2 = module(custom_getter=stop_gradient, **module_kwargs)
    out2 = conv_mod2(inputs)

    num_variables = len(conv_mod1.get_variables())

    grads1 = tf.gradients(out1, list(conv_mod1.get_variables()))
    grads2 = tf.gradients(out2, list(conv_mod2.get_variables()))

    self.assertEqual([tf.Tensor] * num_variables, [type(g) for g in grads1])
    self.assertEqual([None] * num_variables, grads2)

    # Check that the transpose, if present, also adopts the custom getter.
    if hasattr(conv_mod2, "transpose"):
      conv_mod2_transpose = conv_mod2.transpose()
      inputs_transpose = tf.placeholder(tf.float32, out2.get_shape())
      out3 = conv_mod2_transpose(inputs_transpose)
      grads3 = tf.gradients(out3, list(conv_mod2_transpose.get_variables()))
      self.assertEqual([None] * num_variables, grads3)


class Conv2DTest(parameterized.ParameterizedTestCase, tf.test.TestCase):

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testShapesSame(self, use_bias):
    """The generated shapes are correct with SAME padding."""

    batch_size = random.randint(1, 100)
    in_height = random.randint(10, 288)
    in_width = random.randint(10, 288)
    in_channels = random.randint(1, 10)
    out_channels = random.randint(1, 32)
    kernel_shape_h = random.randint(1, 11)
    kernel_shape_w = random.randint(1, 11)

    inputs = tf.placeholder(
        tf.float32,
        shape=[batch_size, in_height, in_width, in_channels])

    conv1 = snt.Conv2D(
        name="conv1",
        output_channels=out_channels,
        kernel_shape=[kernel_shape_h, kernel_shape_w],
        padding=snt.SAME,
        stride=1,
        use_bias=use_bias)

    output = conv1(inputs)

    self.assertTrue(
        output.get_shape().is_compatible_with(
            [batch_size, in_height, in_width, out_channels]))

    self.assertTrue(
        conv1.w.get_shape().is_compatible_with(
            [kernel_shape_h, kernel_shape_w, in_channels, out_channels]))

    if use_bias:
      self.assertTrue(
          conv1.b.get_shape().is_compatible_with(
              [out_channels]))

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testShapesNotKnown(self, use_bias):
    """The generated shapes are correct when input shape not known."""

    batch_size = 5
    in_height = in_width = 32
    in_channels = out_channels = 5
    kernel_shape_h = kernel_shape_w = 3

    inputs = tf.placeholder(
        tf.float32,
        shape=[None, None, None, in_channels],
        name="inputs")

    conv1 = snt.Conv2D(
        name="conv1",
        output_channels=out_channels,
        kernel_shape=[kernel_shape_h, kernel_shape_w],
        padding=snt.SAME,
        stride=1,
        use_bias=use_bias)

    output = conv1(inputs)

    with self.test_session():
      tf.variables_initializer(
          [conv1.w, conv1.b] if use_bias else [conv1.w]).run()

      output_eval = output.eval({
          inputs: np.zeros([batch_size, in_height, in_width, in_channels])})

      self.assertEqual(
          output_eval.shape,
          (batch_size, in_height, in_width, out_channels))

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testShapesNotKnownAtrous(self, use_bias):
    """No error is thrown if image shape isn't known for atrous convolution."""

    inputs = tf.placeholder(
        tf.float32,
        shape=[None, None, None, 5],
        name="inputs")

    conv1 = snt.Conv2D(
        name="conv1",
        output_channels=5,
        kernel_shape=[3, 3],
        padding=snt.SAME,
        stride=1,
        rate=2,
        use_bias=use_bias)

    conv1(inputs)

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testKernelShape(self, use_bias):
    """Errors are thrown for invalid kernel shapes."""

    snt.Conv2D(output_channels=10, kernel_shape=[3, 4], name="conv1",
               use_bias=use_bias)
    snt.Conv2D(output_channels=10, kernel_shape=3, name="conv1",
               use_bias=use_bias)

    err = "Invalid kernel shape"
    with self.assertRaisesRegexp(snt.IncompatibleShapeError, err):
      snt.Conv2D(output_channels=10,
                 kernel_shape=[3, 3, 3],
                 name="conv1")

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testStrideError(self, use_bias):
    """Errors are thrown for invalid strides."""

    snt.Conv2D(
        output_channels=10, kernel_shape=3, stride=1, name="conv1",
        use_bias=use_bias)
    snt.Conv2D(
        output_channels=10, kernel_shape=3, stride=[1, 1], name="conv1",
        use_bias=use_bias)
    snt.Conv2D(
        output_channels=10, kernel_shape=3, stride=[1, 1, 1, 1], name="conv1",
        use_bias=use_bias)

    with self.assertRaisesRegexp(snt.IncompatibleShapeError, "Invalid stride"):
      snt.Conv2D(output_channels=10,
                 kernel_shape=3,
                 stride=[1, 1, 1],
                 name="conv1")

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testRateError(self, use_bias):
    """Errors are thrown for invalid dilation rates."""

    snt.Conv2D(
        output_channels=10, kernel_shape=3, rate=1, name="conv1",
        use_bias=use_bias)
    snt.Conv2D(
        output_channels=10, kernel_shape=3, rate=2, name="conv1",
        use_bias=use_bias)

    for rate in [0, 0.5, -1]:
      with self.assertRaisesRegexp(snt.IncompatibleShapeError,
                                   "Invalid rate shape*"):
        snt.Conv2D(output_channels=10,
                   kernel_shape=3,
                   rate=rate,
                   name="conv1")

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testRateAndStrideError(self, use_bias):
    """Errors are thrown for stride > 1 when using atrous convolution."""
    err = "Cannot have stride > 1 with rate > 1"
    with self.assertRaisesRegexp(snt.NotSupportedError, err):
      snt.Conv2D(output_channels=10, kernel_shape=3,
                 stride=2, rate=2, name="conv1", use_bias=use_bias)
    with self.assertRaisesRegexp(snt.NotSupportedError, err):
      snt.Conv2D(output_channels=10, kernel_shape=3,
                 stride=[2, 1], rate=2, name="conv1", use_bias=use_bias)

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testInputTypeError(self, use_bias):
    """Errors are thrown for invalid input types."""
    conv1 = snt.Conv2D(output_channels=1,
                       kernel_shape=3,
                       stride=1,
                       padding=snt.SAME,
                       name="conv1",
                       use_bias=use_bias,
                       initializers=create_constant_initializers(
                           1.0, 1.0, use_bias))

    for dtype in (tf.float16, tf.float64):
      x = tf.constant(np.ones([1, 5, 5, 1]), dtype=dtype)
      err = "Input must have dtype tf.float32.*"
      with self.assertRaisesRegexp(TypeError, err):
        conv1(x)

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testInitializers(self, use_bias):
    """Test initializers work as expected."""
    w = random.random()
    b = random.random()

    conv1 = snt.Conv2D(
        output_channels=1,
        kernel_shape=3,
        stride=1,
        name="conv1",
        use_bias=use_bias,
        initializers=create_constant_initializers(w, b, use_bias))

    conv1(tf.placeholder(tf.float32, [1, 10, 10, 2]))

    with self.test_session():
      tf.variables_initializer(
          [conv1.w, conv1.b] if use_bias else [conv1.w]).run()

      self.assertAllClose(
          conv1.w.eval(),
          np.full([3, 3, 2, 1], w, dtype=np.float32))

      if use_bias:
        self.assertAllClose(
            conv1.b.eval(),
            [b])

    err = "Initializer for 'w' is not a callable function or dictionary"
    with self.assertRaisesRegexp(TypeError, err):
      snt.Conv2D(output_channels=10, kernel_shape=3, stride=1, name="conv1",
                 initializers={"w": tf.ones([])})

  def testInitializerMutation(self):
    """Test that initializers are not mutated."""

    initializers = {"b": tf.constant_initializer(0)}
    initializers_copy = dict(initializers)

    conv1 = snt.Conv2D(
        output_channels=1,
        kernel_shape=3,
        stride=1,
        name="conv1",
        initializers=initializers)

    conv1(tf.placeholder(tf.float32, [1, 10, 10, 2]))

    self.assertAllEqual(initializers, initializers_copy)

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testRegularizersInRegularizationLosses(self, use_bias):
    regularizers = create_regularizers(
        use_bias, tf.contrib.layers.l1_regularizer(scale=0.5))

    conv1 = snt.Conv2D(
        output_channels=1,
        kernel_shape=3,
        stride=1,
        regularizers=regularizers,
        use_bias=use_bias,
        name="conv1")
    conv1(tf.placeholder(tf.float32, [1, 10, 10, 2]))

    graph_regularizers = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    self.assertRegexpMatches(graph_regularizers[0].name, ".*l1_regularizer.*")
    if use_bias:
      self.assertRegexpMatches(graph_regularizers[1].name, ".*l1_regularizer.*")

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testComputationSame(self, use_bias):
    """Run through for something with a known answer using SAME padding."""
    conv1 = snt.Conv2D(
        output_channels=1,
        kernel_shape=3,
        stride=1,
        padding=snt.SAME,
        name="conv1",
        use_bias=use_bias,
        initializers=create_constant_initializers(1.0, 1.0, use_bias))

    out = conv1(tf.constant(np.ones([1, 5, 5, 1], dtype=np.float32)))
    expected_out = np.array([[5, 7, 7, 7, 5],
                             [7, 10, 10, 10, 7],
                             [7, 10, 10, 10, 7],
                             [7, 10, 10, 10, 7],
                             [5, 7, 7, 7, 5]])
    if not use_bias:
      expected_out -= 1

    with self.test_session():
      tf.variables_initializer(
          [conv1.w, conv1.b] if use_bias else [conv1.w]).run()

      self.assertAllClose(np.reshape(out.eval(), [5, 5]), expected_out)

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testComputationValid(self, use_bias):
    """Run through for something with a known answer using snt.VALID padding."""
    conv1 = snt.Conv2D(
        output_channels=1,
        kernel_shape=3,
        stride=1,
        padding=snt.VALID,
        name="conv1",
        use_bias=use_bias,
        initializers=create_constant_initializers(1.0, 1.0, use_bias))

    out = conv1(tf.constant(np.ones([1, 5, 5, 1], dtype=np.float32)))
    expected_output = np.array([[10, 10, 10],
                                [10, 10, 10],
                                [10, 10, 10]])
    if not use_bias:
      expected_output -= 1

    with self.test_session():
      tf.variables_initializer(
          [conv1.w, conv1.b] if use_bias else [conv1.w]).run()

      self.assertAllClose(np.reshape(out.eval(), [3, 3]), expected_output)

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testSharing(self, use_bias):
    """Sharing is working."""

    conv1 = snt.Conv2D(
        output_channels=1,
        kernel_shape=3,
        stride=1,
        padding=snt.SAME,
        use_bias=use_bias,
        name="conv1")

    x = np.random.randn(1, 5, 5, 1)
    x1 = tf.constant(x, dtype=np.float32)
    x2 = tf.constant(x, dtype=np.float32)

    out1 = conv1(x1)
    out2 = conv1(x2)

    with self.test_session():
      tf.variables_initializer(
          [conv1.w, conv1.b] if use_bias else [conv1.w]).run()

      self.assertAllClose(
          out1.eval(),
          out2.eval())

      # Now change the weights
      w = np.random.randn(3, 3, 1, 1)
      conv1.w.assign(w).eval()

      self.assertAllClose(
          out1.eval(),
          out2.eval())

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testAtrousConvValid(self, use_bias):
    """The atrous conv is constructed and applied correctly with snt.VALID."""
    conv1 = snt.Conv2D(
        output_channels=1,
        kernel_shape=3,
        stride=1,
        rate=2,
        padding=snt.VALID,
        name="conv1",
        use_bias=use_bias,
        initializers=create_constant_initializers(1.0, 0.0, use_bias))

    out = conv1(tf.constant(np.ones([1, 5, 5, 1], dtype=np.float32)))

    with self.test_session():
      tf.variables_initializer(
          [conv1.w, conv1.b] if use_bias else [conv1.w]).run()

      self.assertAllClose(np.reshape(out.eval(), [1, 1]), [[9]])

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testAtrousConvSame(self, use_bias):
    """The atrous conv 2D is constructed and applied correctly with SAME."""
    conv1 = snt.Conv2D(
        output_channels=1,
        kernel_shape=3,
        stride=1,
        rate=2,
        padding=snt.SAME,
        name="conv1",
        use_bias=use_bias,
        initializers=create_constant_initializers(1.0, 1.0, use_bias))

    out = conv1(tf.constant(np.ones([1, 5, 5, 1], dtype=np.float32)))
    expected_out = np.array([[5, 5, 7, 5, 5],
                             [5, 5, 7, 5, 5],
                             [7, 7, 10, 7, 7],
                             [5, 5, 7, 5, 5],
                             [5, 5, 7, 5, 5]])
    if not use_bias:
      expected_out -= 1

    with self.test_session():
      tf.variables_initializer(
          [conv1.w, conv1.b] if use_bias else [conv1.w]).run()

      self.assertAllClose(np.reshape(out.eval(), [5, 5]), expected_out)

  def testClone(self):
    net = snt.Conv2D(name="conv2d",
                     output_channels=4,
                     kernel_shape=3,
                     stride=5)
    clone1 = net.clone()
    clone2 = net.clone(name="clone2")

    input_to_net = tf.placeholder(tf.float32, shape=[None, 100, 100, 3])
    net_out = net(input_to_net)
    clone1_out = clone1(input_to_net)
    clone2_out = clone2(input_to_net)

    all_vars = tf.trainable_variables()
    net_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES,
        scope=net.variable_scope.name + "/")
    clone1_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES,
        scope=clone1.variable_scope.name + "/")
    clone2_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES,
        scope=clone2.variable_scope.name + "/")

    self.assertEqual(net.output_channels, clone1.output_channels)
    self.assertEqual(net.module_name + "_clone", clone1.module_name)
    self.assertEqual("clone2", clone2.module_name)
    self.assertEqual(len(all_vars), 3*len(net_vars))
    self.assertEqual(len(net_vars), len(clone1_vars))
    self.assertEqual(len(net_vars), len(clone2_vars))
    self.assertEqual(net_out.get_shape().as_list(),
                     clone1_out.get_shape().as_list())
    self.assertEqual(net_out.get_shape().as_list(),
                     clone2_out.get_shape().as_list())

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testTransposition(self, use_bias):
    """Tests if the correct output shapes are setup in transposed module."""
    net = snt.Conv2D(name="conv2d",
                     output_channels=4,
                     kernel_shape=3,
                     stride=1,
                     use_bias=use_bias)

    net_transpose = net.transpose()
    input_to_net = tf.placeholder(tf.float32, shape=[None, 100, 100, 3])

    err = "Variables in {} not instantiated yet, __call__ the module first."
    with self.assertRaisesRegexp(snt.NotConnectedError,
                                 err.format(net.scope_name)):
      net_transpose(input_to_net)
    net_transpose = net.transpose(name="another_net_transpose")
    net_out = net(input_to_net)
    net_transposed_output = net_transpose(net_out)
    self.assertAllEqual(net_transposed_output.get_shape().as_list(),
                        input_to_net.get_shape().as_list())

  def testMask2D(self):
    """2D Masks are applied properly."""

    # This mask, applied on an image filled with 1, should result in an image
    # filled with 8 (since we sum 4 elements per channel and there are 2 input
    # channels).
    mask = np.array([[1, 1, 1],
                     [1, 0, 0],
                     [0, 0, 0]], dtype=np.float32)
    inputs = tf.constant(1.0, shape=(1, 5, 5, 2))
    conv1 = snt.Conv2D(
        output_channels=1,
        kernel_shape=3,
        mask=mask,
        padding=snt.VALID,
        use_bias=False,
        initializers=create_constant_initializers(1.0, 0.0, use_bias=False))
    out = conv1(inputs)
    expected_out = np.array([[8] * 3] * 3)
    with self.test_session():
      tf.variables_initializer([conv1.w]).run()
      self.assertAllClose(np.reshape(out.eval(), [3, 3]), expected_out)

  def testMask4D(self):
    """4D Masks are applied properly."""

    # This mask, applied on an image filled with 1, should result in an image
    # filled with 17, as there are 18 weights but we zero out one of them.
    mask = np.ones([3, 3, 2, 1], dtype=np.float32)
    mask[0, 0, 0, :] = 0
    inputs = tf.constant(1.0, shape=(1, 5, 5, 2))
    conv1 = snt.Conv2D(
        output_channels=1,
        kernel_shape=3,
        mask=mask,
        padding=snt.VALID,
        use_bias=False,
        initializers=create_constant_initializers(1.0, 0.0, use_bias=False))
    out = conv1(inputs)
    expected_out = np.array([[17] * 3] * 3)
    with self.test_session():
      tf.variables_initializer([conv1.w]).run()
      self.assertAllClose(np.reshape(out.eval(), [3, 3]), expected_out)

  def testMaskErrorInvalidRank(self):
    """Errors are thrown for invalid mask rank."""

    mask = np.ones((3,))
    with self.assertRaises(snt.Error) as cm:
      snt.Conv2D(output_channels=4, kernel_shape=3, mask=mask)
    self.assertEqual(
        str(cm.exception),
        "Invalid mask rank: {}".format(mask.ndim))

  def testMaskErrorInvalidType(self):
    """Errors are thrown for invalid mask type."""

    mask = tf.constant(1.0, shape=(3, 3))
    with self.assertRaises(TypeError) as cm:
      snt.Conv2D(output_channels=4, kernel_shape=3, mask=mask)
    self.assertEqual(
        str(cm.exception), "Invalid type for mask: {}".format(type(mask)))

  def testMaskErrorIncompatibleRank2(self):
    """Errors are thrown for incompatible rank 2 mask."""

    mask = np.ones((3, 3))
    x = tf.constant(0.0, shape=(2, 8, 8, 6))
    with self.assertRaises(snt.Error) as cm:
      snt.Conv2D(output_channels=4, kernel_shape=5, mask=mask)(x)
    self.assertTrue(str(cm.exception).startswith(
        "Invalid mask shape: {}".format(mask.shape)))

  def testMaskErrorIncompatibleRank4(self):
    """Errors are thrown for incompatible rank 4 mask."""

    mask = np.ones((3, 3, 4, 5))
    x = tf.constant(0.0, shape=(2, 8, 8, 6))
    with self.assertRaises(snt.Error) as cm:
      snt.Conv2D(output_channels=4, kernel_shape=5, mask=mask)(x)
    self.assertTrue(str(cm.exception).startswith(
        "Invalid mask shape: {}".format(mask.shape)))


class Conv2DTransposeTest(parameterized.ParameterizedTestCase,
                          tf.test.TestCase):

  def setUp(self):
    """Set up some variables to re-use in multiple tests."""

    super(Conv2DTransposeTest, self).setUp()

    self.batch_size = 100
    self.in_height = 32
    self.in_width = 32
    self.in_channels = 3
    self.out_channels = 10
    self.kernel_shape_h = 5
    self.kernel_shape_w = 5
    self.strides = (1, 1, 1, 1)
    self.padding = snt.SAME

    self.in_shape = (self.batch_size, self.in_height, self.in_width,
                     self.in_channels)

    self.out_shape = (self.in_height, self.in_width)

    self.kernel_shape = (self.kernel_shape_h, self.kernel_shape_w)

    self.kernel_shape2 = (self.kernel_shape_h, self.kernel_shape_w,
                          self.out_channels, self.in_channels)

  def testKernelsNotSpecified(self):
    """Tests error is raised if kernel shape is not specified."""
    with self.assertRaisesRegexp(ValueError, "`kernel_shape` cannot be None."):
      snt.Conv2DTranspose(output_channels=1)

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testOutputShapeConsistency(self, use_bias):
    """Tests if output shapes are valid."""

    # When padding is SAME, then the actual number of padding pixels can be
    # computed as: pad = kernel_shape - strides + (-input_shape % strides)
    #                 =     5         -    1    + (- 32       %      1) = 4

    # The formula for the minimal size is:
    # oH = strides[1] * (in_height - 1) - padding + kernel_shape_h
    # oH =          1 * (       32 - 1) -    4    +       5 = 32

    # The formula for the maximum size (due to extra pixels) is:
    # oH_max = oH + strides[1] - 1
    # so, for strides = 1 and padding = SAME, input size == output size.
    inputs = tf.placeholder(tf.float32, shape=self.in_shape)

    conv1 = snt.Conv2DTranspose(name="conv2d_1",
                                output_channels=self.out_channels,
                                output_shape=self.out_shape,
                                kernel_shape=self.kernel_shape,
                                padding=self.padding,
                                stride=1,
                                use_bias=use_bias)

    outputs = conv1(inputs)

    self.assertTrue(outputs.get_shape().is_compatible_with((
        self.batch_size,) + self.out_shape + (self.out_channels,)))

    self.assertTrue(conv1.w.get_shape().is_compatible_with(self.kernel_shape2))
    if use_bias:
      self.assertTrue(conv1.b.get_shape().is_compatible_with(
          [self.out_channels]))

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testOutputShapeInteger(self, use_bias):
    """Tests if output shapes are valid when specified as an integer."""
    inputs = tf.zeros(shape=[3, 5, 5, 2], dtype=tf.float32)
    inputs_2 = tf.zeros(shape=[3, 5, 7, 2], dtype=tf.float32)

    conv1 = snt.Conv2DTranspose(name="conv2d_1",
                                output_channels=10,
                                output_shape=10,
                                kernel_shape=5,
                                padding=snt.SAME,
                                stride=2,
                                use_bias=use_bias)

    outputs = conv1(inputs)
    outputs_2 = conv1(inputs_2)

    self.assertTrue(outputs.get_shape().is_compatible_with((3, 10, 10, 10)))

    with self.test_session() as sess:
      tf.global_variables_initializer().run()
      sess.run(outputs)
      with self.assertRaises(tf.errors.InvalidArgumentError):
        sess.run(outputs_2)

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testTransposition(self, use_bias):
    """Tests if the correct output shapes are setup in transposed module."""
    net = snt.Conv2DTranspose(name="conv2d",
                              output_channels=self.out_channels,
                              output_shape=self.out_shape,
                              kernel_shape=self.kernel_shape,
                              padding=self.padding,
                              stride=1,
                              use_bias=use_bias)

    net_transpose = net.transpose()
    input_to_net = tf.placeholder(tf.float32, shape=self.in_shape)
    err = "Variables in {} not instantiated yet, __call__ the module first."
    with self.assertRaisesRegexp(snt.NotConnectedError,
                                 err.format(net.scope_name)):
      net_transpose(input_to_net)
    net_transpose = net.transpose(name="another_net_transpose")
    net_out = net(input_to_net)
    net_transposed_output = net_transpose(net_out)
    self.assertEqual(net_transposed_output.get_shape(),
                     input_to_net.get_shape())

  def testInitializerMutation(self):
    """Test that initializers are not mutated."""

    initializers = {"b": tf.constant_initializer(0)}
    initializers_copy = dict(initializers)

    conv1 = snt.Conv2DTranspose(
        output_shape=(10, 10),
        output_channels=1,
        kernel_shape=3,
        stride=1,
        name="conv2d",
        initializers=initializers)

    conv1(tf.placeholder(tf.float32, [1, 10, 10, 2]))

    self.assertAllEqual(initializers, initializers_copy)


class Conv1DTest(parameterized.ParameterizedTestCase, tf.test.TestCase):

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testShapes(self, use_bias):
    """The generated shapes are correct with SAME and VALID padding."""

    batch_size = random.randint(1, 100)
    in_length = random.randint(10, 288)
    in_channels = random.randint(1, 10)
    out_channels = random.randint(1, 32)

    kernel_shape = random.randint(1, 10)

    inputs = tf.placeholder(
        tf.float32,
        shape=[batch_size, in_length, in_channels])

    conv1 = snt.Conv1D(
        output_channels=out_channels,
        kernel_shape=kernel_shape,
        padding=snt.SAME,
        stride=1,
        name="conv1",
        use_bias=use_bias)

    output1 = conv1(inputs)

    self.assertTrue(
        output1.get_shape().is_compatible_with(
            [batch_size, in_length, out_channels]))

    self.assertTrue(
        conv1.w.get_shape().is_compatible_with(
            [kernel_shape, in_channels, out_channels]))

    if use_bias:
      self.assertTrue(
          conv1.b.get_shape().is_compatible_with(
              [out_channels]))

    conv2 = snt.Conv1D(
        output_channels=out_channels,
        kernel_shape=kernel_shape,
        padding=snt.VALID,
        stride=1,
        name="conv2",
        use_bias=use_bias)

    output2 = conv2(inputs)

    self.assertTrue(
        output2.get_shape().is_compatible_with(
            [batch_size, in_length - kernel_shape + 1, out_channels]))

    self.assertTrue(
        conv2.w.get_shape().is_compatible_with(
            [kernel_shape, in_channels, out_channels]))

    if use_bias:
      self.assertTrue(
          conv2.b.get_shape().is_compatible_with(
              [out_channels]))

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testShapesNotKnown(self, use_bias):
    """The generated shapes are correct when input shape not known."""

    batch_size = 5
    in_length = 32
    in_channels = out_channels = 5
    kernel_shape = 3

    inputs = tf.placeholder(
        tf.float32,
        shape=[None, None, in_channels],
        name="inputs")

    conv1 = snt.Conv1D(
        name="conv1",
        output_channels=out_channels,
        kernel_shape=kernel_shape,
        padding=snt.SAME,
        stride=1,
        use_bias=use_bias)

    output = conv1(inputs)

    with self.test_session():
      tf.variables_initializer(
          [conv1.w, conv1.b] if use_bias else [conv1.w]).run()

      output_eval = output.eval({
          inputs: np.zeros([batch_size, in_length, in_channels])})

      self.assertEqual(
          output_eval.shape,
          (batch_size, in_length, out_channels))

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testKernelShape(self, use_bias):
    """Errors are thrown for invalid kernel shapes."""

    snt.Conv1D(output_channels=10, kernel_shape=[3], name="conv1",
               use_bias=use_bias)
    snt.Conv1D(output_channels=10, kernel_shape=3, name="conv1",
               use_bias=use_bias)

    err = "Invalid kernel shape"
    with self.assertRaisesRegexp(snt.IncompatibleShapeError, err):
      snt.Conv1D(output_channels=10, kernel_shape=[3, 3], name="conv1")

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testStrideError(self, use_bias):
    """Errors are thrown for invalid strides."""

    snt.Conv1D(
        output_channels=10, kernel_shape=3, stride=1, name="conv1",
        use_bias=use_bias)

    err = "Invalid stride"
    with self.assertRaisesRegexp(snt.IncompatibleShapeError, err):
      snt.Conv1D(output_channels=10, kernel_shape=3,
                 stride=[1, 1], name="conv1")

    with self.assertRaisesRegexp(snt.IncompatibleShapeError, err):
      snt.Conv1D(output_channels=10, kernel_shape=3,
                 stride=[1, 1, 1, 1], name="conv1")

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testRateError(self, use_bias):
    """Errors are thrown for invalid dilation rates."""

    snt.Conv1D(
        output_channels=10, kernel_shape=3, rate=1, name="conv1",
        use_bias=use_bias)
    snt.Conv1D(
        output_channels=10, kernel_shape=3, rate=2, name="conv1",
        use_bias=use_bias)

    for rate in [0, 0.5, -1]:
      with self.assertRaisesRegexp(snt.IncompatibleShapeError,
                                   "Invalid rate shape*"):
        snt.Conv1D(output_channels=10,
                   kernel_shape=3,
                   rate=rate,
                   name="conv1")

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testRateAndStrideError(self, use_bias):
    """Errors are thrown for stride > 1 when using atrous convolution."""
    err = "Cannot have stride > 1 with rate > 1"
    with self.assertRaisesRegexp(snt.NotSupportedError, err):
      snt.Conv1D(output_channels=10, kernel_shape=3,
                 stride=2, rate=2, name="conv1", use_bias=use_bias)

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testInputTypeError(self, use_bias):
    """Errors are thrown for invalid input types."""
    conv1 = snt.Conv1D(output_channels=1,
                       kernel_shape=3,
                       stride=1,
                       padding=snt.VALID,
                       use_bias=use_bias,
                       name="conv1",
                       initializers=create_constant_initializers(
                           1.0, 1.0, use_bias))

    for dtype in (tf.float16, tf.float64):
      x = tf.constant(np.ones([1, 5, 1]), dtype=dtype)
      err = "Input must have dtype tf.float32.*"
      with self.assertRaisesRegexp(TypeError, err):
        conv1(x)

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testInitializers(self, use_bias):
    """Test initializers work as expected."""
    w = random.random()
    b = random.random()

    conv1 = snt.Conv1D(
        output_channels=1,
        kernel_shape=3,
        stride=1,
        padding=snt.SAME,
        use_bias=use_bias,
        name="conv1",
        initializers=create_constant_initializers(w, b, use_bias))

    conv1(tf.placeholder(tf.float32, [1, 10, 2]))

    with self.test_session():
      tf.variables_initializer(
          [conv1.w, conv1.b] if use_bias else [conv1.w]).run()

      self.assertAllClose(
          conv1.w.eval(),
          np.full([3, 2, 1], w, dtype=np.float32))

      if use_bias:
        self.assertAllClose(
            conv1.b.eval(),
            [b])

    err = "Initializer for 'w' is not a callable function or dictionary"
    with self.assertRaisesRegexp(TypeError, err):
      snt.Conv1D(output_channels=10,
                 kernel_shape=3,
                 stride=1,
                 padding=snt.SAME,
                 use_bias=use_bias,
                 name="conv1",
                 initializers={"w": tf.ones([])})

  def testInitializerMutation(self):
    """Test that initializers are not mutated."""

    initializers = {"b": tf.constant_initializer(0)}
    initializers_copy = dict(initializers)

    conv1 = snt.Conv1D(
        output_channels=1,
        kernel_shape=3,
        stride=1,
        name="conv1",
        initializers=initializers)

    conv1(tf.placeholder(tf.float32, [1, 10, 2]))

    self.assertAllEqual(initializers, initializers_copy)

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testRegularizersInRegularizationLosses(self, use_bias):
    regularizers = create_regularizers(
        use_bias, tf.contrib.layers.l1_regularizer(scale=0.5))

    conv1 = snt.Conv1D(
        output_channels=1,
        kernel_shape=3,
        stride=1,
        regularizers=regularizers,
        name="conv1")
    conv1(tf.placeholder(tf.float32, [1, 10, 2]))

    graph_regularizers = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    self.assertRegexpMatches(graph_regularizers[0].name, ".*l1_regularizer.*")
    if use_bias:
      self.assertRegexpMatches(graph_regularizers[1].name, ".*l1_regularizer.*")

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testComputationSame(self, use_bias):
    """Run through for something with a known answer using SAME padding."""
    conv1 = snt.Conv1D(
        output_channels=1,
        kernel_shape=3,
        stride=1,
        padding=snt.SAME,
        use_bias=use_bias,
        name="conv1",
        initializers=create_constant_initializers(1.0, 1.0, use_bias))

    out = conv1(tf.constant(np.ones([1, 5, 1], dtype=np.float32)))
    expected_out = np.asarray([3, 4, 4, 4, 3])
    if not use_bias:
      expected_out -= 1

    with self.test_session():
      tf.variables_initializer(
          [conv1.w, conv1.b] if use_bias else [conv1.w]).run()

      self.assertAllClose(np.reshape(out.eval(), [5]), expected_out)

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testComputationValid(self, use_bias):
    """Run through for something with a known answer using snt.VALID padding."""
    conv1 = snt.Conv1D(
        output_channels=1,
        kernel_shape=3,
        stride=1,
        padding=snt.VALID,
        use_bias=use_bias,
        name="conv1",
        initializers=create_constant_initializers(1.0, 1.0, use_bias))

    out = conv1(tf.constant(np.ones([1, 5, 1], dtype=np.float32)))
    expected_out = np.asarray([4, 4, 4])
    if not use_bias:
      expected_out -= 1

    with self.test_session():
      tf.variables_initializer(
          [conv1.w, conv1.b] if use_bias else [conv1.w]).run()

      self.assertAllClose(np.reshape(out.eval(), [3]), expected_out)

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testSharing(self, use_bias):
    """Sharing is working."""

    conv1 = snt.Conv1D(
        output_channels=1,
        kernel_shape=3,
        stride=1,
        padding=snt.SAME,
        use_bias=use_bias,
        name="conv1")

    x = np.random.randn(1, 5, 1)
    x1 = tf.constant(x, dtype=np.float32)
    x2 = tf.constant(x, dtype=np.float32)

    out1 = conv1(x1)
    out2 = conv1(x2)

    with self.test_session():
      tf.variables_initializer(
          [conv1.w, conv1.b] if use_bias else [conv1.w]).run()

      self.assertAllClose(
          out1.eval(),
          out2.eval())

      # Now change the weights
      w = np.random.randn(3, 1, 1)
      conv1.w.assign(w).eval()

      self.assertAllClose(
          out1.eval(),
          out2.eval())


class Conv1DTransposeTest(parameterized.ParameterizedTestCase,
                          tf.test.TestCase):

  # Constants for use in all tests.
  batch_size = [10, 2, 8, 18, 23]
  in_length = [20, 23, 24, 15, 16]
  in_channels = [6, 2, 3, 6, 9]
  out_channels = [18, 19, 15, 32, 5]
  kernel_shape = [4, 10, 1, 2, 7]
  stride = [1, 2, 4, 7, 5]
  padding = [snt.SAME, snt.SAME, snt.VALID, snt.VALID, snt.VALID]
  use_bias = [True, False, True, False, True]
  out_length = []

  for i, pad in enumerate(padding):
    if pad == snt.SAME:
      out_length.append(in_length[i] * stride[i])
    if pad == snt.VALID:
      out_length.append(in_length[i] * stride[i] + kernel_shape[i] - 1)

  in_shape = tuple(zip(batch_size, in_length, in_channels))
  out_shape = tuple(out_length)
  kernel_shape = tuple(kernel_shape)
  kernel_shape2 = tuple(zip(kernel_shape, out_channels, in_channels))
  stride_shape = tuple(stride)

  def testKernelsNotSpecified(self):
    """Tests error is raised if kernel shape is not specified."""
    with self.assertRaisesRegexp(ValueError, "`kernel_shape` cannot be None."):
      snt.Conv1DTranspose(output_channels=1)

  @parameterized.Parameters(
      *zip(out_channels, kernel_shape, padding, use_bias, in_shape, out_shape,
           stride_shape))
  def testMissingBatchSize(self, out_channels, kernel_shape, padding,
                           use_bias, in_shape, out_shape, stride_shape):
    """Check functionality with unknown batch size at build time."""

    conv1 = snt.Conv1DTranspose(output_channels=out_channels,
                                output_shape=out_shape,
                                kernel_shape=kernel_shape,
                                padding=padding,
                                stride=stride_shape,
                                name="conv1",
                                use_bias=use_bias)

    # Pass in an image with its batch size set to `None`:
    image = tf.placeholder(tf.float32, shape=(None,) + in_shape[1:])
    output = conv1(image)
    self.assertTrue(output.get_shape().is_compatible_with(
        [None, out_shape, out_channels]))

    with self.test_session() as sess:
      tf.global_variables_initializer().run()
      sess.run(output, feed_dict={image: np.zeros((10,) + in_shape[1:])})

  @parameterized.Parameters(
      *zip(batch_size, in_length, in_channels, out_length, out_channels,
           kernel_shape, padding, use_bias, in_shape, out_shape, stride_shape))
  def testShapesSame(self, batch_size, in_length, in_channels, out_length,
                     out_channels, kernel_shape, padding, use_bias, in_shape,
                     out_shape, stride_shape):
    """The generated shapes are correct."""

    inputs = tf.placeholder(
        tf.float32,
        shape=[batch_size, in_length, in_channels])

    conv1 = snt.Conv1DTranspose(output_channels=out_channels,
                                output_shape=out_shape,
                                kernel_shape=kernel_shape,
                                padding=padding,
                                stride=stride_shape,
                                name="conv1",
                                use_bias=use_bias)

    output = conv1(inputs)

    self.assertTrue(
        output.get_shape().is_compatible_with(
            [batch_size, out_length, out_channels]))

    self.assertTrue(
        conv1.w.get_shape().is_compatible_with(
            [1, kernel_shape, out_channels, in_channels]))

    if use_bias:
      self.assertTrue(
          conv1.b.get_shape().is_compatible_with(
              [out_channels]))

  @parameterized.Parameters(
      *zip(out_channels, padding, use_bias, in_shape, out_shape, stride_shape))
  def testKernelShape(self, out_channels, padding, use_bias, in_shape,
                      out_shape, stride_shape):
    """Errors are thrown for invalid kernel shapes."""

    snt.Conv1DTranspose(
        output_channels=out_channels,
        output_shape=out_shape,
        kernel_shape=[3],
        padding=padding,
        stride=stride_shape,
        name="conv1",
        use_bias=use_bias)
    snt.Conv1DTranspose(
        output_channels=out_channels,
        output_shape=out_shape,
        kernel_shape=3,
        padding=padding,
        stride=stride_shape,
        name="conv1",
        use_bias=use_bias)

    err = "Invalid kernel"
    with self.assertRaisesRegexp(snt.IncompatibleShapeError, err):
      snt.Conv1DTranspose(output_channels=out_channels,
                          output_shape=out_shape,
                          kernel_shape=[3, 3],
                          name="conv1",
                          use_bias=use_bias)

    with self.assertRaisesRegexp(snt.IncompatibleShapeError, err):
      snt.Conv1DTranspose(output_channels=out_channels,
                          output_shape=out_shape,
                          kernel_shape=[3, 3, 3, 3],
                          name="conv1",
                          use_bias=use_bias)

  @parameterized.Parameters(
      *zip(out_channels, padding, use_bias, in_shape, out_shape))
  def testStrideError(self, out_channels, padding, use_bias, in_shape,
                      out_shape):
    """Errors are thrown for invalid strides."""

    snt.Conv1DTranspose(
        output_channels=out_channels,
        output_shape=out_shape,
        kernel_shape=3,
        padding=padding,
        stride=1,
        name="conv1",
        use_bias=use_bias)

    err = ("must be either a positive integer or an iterable of positive "
           "integers of size 1")
    with self.assertRaisesRegexp(snt.IncompatibleShapeError, err):
      snt.Conv1DTranspose(output_channels=out_channels,
                          output_shape=out_shape,
                          kernel_shape=3,
                          padding=padding,
                          stride=[1, 1],
                          name="conv1",
                          use_bias=use_bias)

    with self.assertRaisesRegexp(snt.IncompatibleShapeError, err):
      snt.Conv1DTranspose(output_channels=out_channels,
                          output_shape=out_shape,
                          kernel_shape=3,
                          padding=padding,
                          stride=[1, 1, 1, 1],
                          name="conv1",
                          use_bias=use_bias)

  @parameterized.Parameters(
      *zip(batch_size, in_length, in_channels, out_channels, kernel_shape,
           padding, use_bias, out_shape, stride_shape))
  def testInputTypeError(self, batch_size, in_length, in_channels, out_channels,
                         kernel_shape, padding, use_bias, out_shape,
                         stride_shape):
    """Errors are thrown for invalid input types."""
    conv1 = snt.Conv1DTranspose(
        output_channels=out_channels,
        output_shape=out_shape,
        kernel_shape=kernel_shape,
        padding=padding,
        stride=stride_shape,
        name="conv1",
        use_bias=use_bias)

    for dtype in (tf.float16, tf.float64):
      x = tf.constant(np.ones([batch_size, in_length,
                               in_channels]), dtype=dtype)
      err = "Input must have dtype tf.float32.*"
      with self.assertRaisesRegexp(TypeError, err):
        conv1(x)

  @parameterized.Parameters(
      *zip(batch_size, in_length, in_channels, out_channels, kernel_shape,
           padding, use_bias, out_shape, stride_shape))
  def testSharing(self, batch_size, in_length, in_channels, out_channels,
                  kernel_shape, padding, use_bias, out_shape, stride_shape):
    """Sharing is working."""

    conv1 = snt.Conv1DTranspose(
        output_channels=out_channels,
        output_shape=out_shape,
        kernel_shape=kernel_shape,
        padding=padding,
        stride=stride_shape,
        name="conv1",
        use_bias=use_bias)

    x = np.random.randn(batch_size, in_length, in_channels)
    x1 = tf.constant(x, dtype=np.float32)
    x2 = tf.constant(x, dtype=np.float32)

    out1 = conv1(x1)
    out2 = conv1(x2)

    with self.test_session():
      tf.variables_initializer(
          [conv1.w, conv1.b] if use_bias else [conv1.w]).run()

      self.assertAllClose(
          out1.eval(),
          out2.eval())

      # Now change the weights
      w = np.random.randn(1, kernel_shape, out_channels, in_channels)
      conv1.w.assign(w).eval()

      self.assertAllClose(
          out1.eval(),
          out2.eval())

  @parameterized.Parameters(
      *zip(batch_size, in_length, in_channels, out_channels, kernel_shape,
           padding, use_bias, out_shape, stride_shape))
  def testTranspose(self, batch_size, in_length, in_channels, out_channels,
                    kernel_shape, padding, use_bias, out_shape, stride_shape):
    """Test transpose."""

    conv1_transpose = snt.Conv1DTranspose(
        output_channels=out_channels,
        output_shape=out_shape,
        kernel_shape=kernel_shape,
        padding=padding,
        stride=stride_shape,
        name="conv1_transpose",
        use_bias=use_bias)
    conv1 = conv1_transpose.transpose()

    # Check kernel shapes, strides and padding match.
    self.assertEqual(conv1_transpose.kernel_shape, conv1.kernel_shape)
    self.assertEqual((1, conv1_transpose.stride[2], 1), conv1.stride)
    self.assertEqual(conv1_transpose.padding, conv1.padding)

    # Before conv1_transpose is connected, we cannot know how many
    # `output_channels` conv1 should have.
    err = "Variables in conv1_transpose not instantiated yet"
    with self.assertRaisesRegexp(snt.NotConnectedError, err):
      conv1.output_channels  # pylint: disable=pointless-statement

    # After connection the number of `output_channels` is known.
    x = tf.constant(np.random.randn(batch_size, in_length, in_channels),
                    dtype=np.float32)
    conv1_transpose(x)
    self.assertEqual(in_channels, conv1.output_channels)

    # However, even after connection, the `input_shape` of the forward
    # convolution is not known until it is itself connected (i.e. it can be
    # connected to a different shape input from the `output_shape` of the
    # transpose convolution!)
    err = "Variables in conv1_transpose_transpose not instantiated yet"
    with self.assertRaisesRegexp(snt.NotConnectedError, err):
      self.assertEqual(conv1_transpose.output_shape, conv1.input_shape)

  def testInitializerMutation(self):
    """Test that initializers are not mutated."""

    initializers = {"b": tf.constant_initializer(0)}
    initializers_copy = dict(initializers)

    conv1 = snt.Conv1DTranspose(
        output_shape=(10,),
        output_channels=1,
        kernel_shape=3,
        stride=1,
        name="conv1",
        initializers=initializers)

    conv1(tf.placeholder(tf.float32, [1, 10, 2]))

    self.assertAllEqual(initializers, initializers_copy)


class CausalConv1DTest(parameterized.ParameterizedTestCase, tf.test.TestCase):

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testComputation(self, use_bias):
    """Run through for something with a known answer."""
    conv1 = snt.CausalConv1D(
        output_channels=1,
        kernel_shape=3,
        stride=1,
        use_bias=use_bias,
        name="conv1",
        initializers=create_constant_initializers(1.0, 1.0, use_bias))

    out = conv1(tf.constant(np.ones([1, 5, 1], dtype=np.float32)))
    expected_out = np.reshape(np.array([1, 2, 3, 3, 3]), [1, 5, 1])
    if use_bias:
      expected_out += 1

    init_op = tf.variables_initializer(
        [conv1.w, conv1.b] if use_bias else [conv1.w])
    with self.test_session() as sess:
      sess.run(init_op)
      actual_out = sess.run(out)

    self.assertAllClose(actual_out, expected_out)

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testComputationStrided(self, use_bias):
    """Run through for something with a known answer."""
    conv1 = snt.CausalConv1D(
        output_channels=1,
        kernel_shape=3,
        stride=2,
        use_bias=use_bias,
        name="conv1",
        initializers=create_constant_initializers(1.0, 1.0, use_bias))

    out = conv1(tf.constant(np.ones([1, 5, 1], dtype=np.float32)))
    expected_out = np.reshape(np.array([1, 3, 3]), [1, 3, 1])
    if use_bias:
      expected_out += 1

    init_op = tf.variables_initializer(
        [conv1.w, conv1.b] if use_bias else [conv1.w])
    with self.test_session() as sess:
      sess.run(init_op)
      actual_out = sess.run(out)

    self.assertAllClose(actual_out, expected_out)

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testComputationDilated(self, use_bias):
    """Run through for something with a known answer."""
    conv1 = snt.CausalConv1D(
        output_channels=1,
        kernel_shape=3,
        stride=1,
        rate=2,
        use_bias=use_bias,
        name="conv1",
        initializers=create_constant_initializers(1.0, 1.0, use_bias))

    out = conv1(tf.constant(np.ones([1, 5, 1], dtype=np.float32)))
    expected_out = np.reshape(np.array([1, 1, 2, 2, 3]), [1, 5, 1])
    if use_bias:
      expected_out += 1

    init_op = tf.variables_initializer(
        [conv1.w, conv1.b] if use_bias else [conv1.w])
    with self.test_session() as sess:
      sess.run(init_op)
      actual_out = sess.run(out)

    self.assertAllClose(actual_out, expected_out)

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testSharing(self, use_bias):
    """Sharing is working."""

    conv1 = snt.CausalConv1D(
        output_channels=1,
        kernel_shape=3,
        stride=1,
        use_bias=use_bias,
        name="conv1")

    x = np.random.randn(1, 5, 1)
    x1 = tf.constant(x, dtype=np.float32)
    x2 = tf.constant(x, dtype=np.float32)

    out1 = conv1(x1)
    out2 = conv1(x2)

    w = np.random.randn(3, 1, 1)
    weight_change_op = conv1.w.assign(w)

    init_op = tf.variables_initializer(
        [conv1.w, conv1.b] if use_bias else [conv1.w])

    with self.test_session() as sess:
      sess.run(init_op)
      first_replica_out = sess.run(out1)
      second_replica_out = sess.run(out2)

      # Now change the weights
      sess.run(weight_change_op)

      first_replica_out_changed = sess.run(out1)
      second_replica_out_changed = sess.run(out2)

    self.assertAllClose(first_replica_out, second_replica_out)
    self.assertAllClose(first_replica_out_changed, second_replica_out_changed)


class InPlaneConv2DTest(parameterized.ParameterizedTestCase, tf.test.TestCase):

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testSameNumberOfOutputAndInputChannels(self, use_bias):
    """Test that the number of output and input channels are equal."""

    input_channels = random.randint(1, 32)
    inputs = tf.placeholder(tf.float32, shape=[1, 10, 10, input_channels])
    conv1 = snt.InPlaneConv2D(kernel_shape=3, use_bias=use_bias)

    # Before conv1 is connected, we cannot know how many `output_channels`
    # conv1 should have.
    err = "Variables in in_plane_conv2d not instantiated yet"
    with self.assertRaisesRegexp(snt.NotConnectedError, err):
      _ = conv1.output_channels

    # After connection, should match `input_channels`.
    conv1(inputs)
    self.assertEqual(conv1.output_channels, input_channels)

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testSharing(self, use_bias):
    """Sharing is working."""

    conv1 = snt.InPlaneConv2D(kernel_shape=3, use_bias=use_bias)
    x = np.random.randn(1, 5, 5, 1)
    x1 = tf.constant(x, dtype=np.float32)
    x2 = tf.constant(x, dtype=np.float32)
    out1 = conv1(x1)
    out2 = conv1(x2)

    with self.test_session():
      tf.variables_initializer(
          [conv1.w, conv1.b] if use_bias else [conv1.w]).run()
      self.assertAllClose(out1.eval(), out2.eval())

      w = np.random.randn(3, 3, 1, 1)  # Now change the weights.
      conv1.w.assign(w).eval()
      self.assertAllClose(out1.eval(), out2.eval())

  def testInitializerMutation(self):
    """Test that initializers are not mutated."""

    initializers = {"b": tf.constant_initializer(0)}
    initializers_copy = dict(initializers)

    conv1 = snt.InPlaneConv2D(kernel_shape=3, initializers=initializers)

    conv1(tf.placeholder(tf.float32, [1, 10, 10, 2]))

    self.assertAllEqual(initializers, initializers_copy)


class DepthwiseConv2DTest(parameterized.ParameterizedTestCase,
                          tf.test.TestCase):

  def setUp(self):
    """Set up some variables to re-use in multiple tests."""

    super(DepthwiseConv2DTest, self).setUp()

    self.batch_size = batch_size = random.randint(1, 20)
    self.in_height = in_height = random.randint(10, 128)
    self.in_width = in_width = random.randint(10, 128)
    self.in_channels = in_channels = random.randint(1, 10)
    self.kernel_shape_h = kernel_shape_h = random.randint(1, 11)
    self.kernel_shape_w = kernel_shape_w = random.randint(1, 11)
    self.channel_multiplier = channel_multiplier = random.randint(1, 10)
    self.out_channels = out_channels = in_channels * channel_multiplier

    self.input_shape = [batch_size, in_height, in_width, in_channels]
    self.kernel_shape = [kernel_shape_h, kernel_shape_w]
    self.output_shape = [batch_size, in_height, in_width, out_channels]
    self.weight_shape = [kernel_shape_h, kernel_shape_w, in_channels,
                         channel_multiplier]

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testShapesSame(self, use_bias):
    """Test that the generated shapes are correct with SAME padding."""

    out_channels = self.out_channels
    input_shape = self.input_shape
    kernel_shape = self.kernel_shape
    output_shape = self.output_shape
    weight_shape = self.weight_shape
    channel_multiplier = self.channel_multiplier

    inputs = tf.placeholder(tf.float32, shape=input_shape)

    conv1 = snt.DepthwiseConv2D(
        name="conv1",
        channel_multiplier=channel_multiplier,
        kernel_shape=kernel_shape,
        padding=snt.SAME,
        stride=1,
        use_bias=use_bias)
    output = conv1(inputs)

    self.assertEqual(output.get_shape(), output_shape)
    self.assertEqual(conv1.w.get_shape(), weight_shape)
    if use_bias:
      self.assertEqual(conv1.b.get_shape(), out_channels)

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testShapesNotKnown(self, use_bias):
    """Test that the generated shapes are correct when input shape not known."""

    inputs = tf.placeholder(
        tf.float32, shape=[None, None, None, self.in_channels], name="inputs")

    conv1 = snt.DepthwiseConv2D(
        channel_multiplier=self.channel_multiplier,
        kernel_shape=self.kernel_shape,
        padding=snt.SAME,
        stride=1,
        use_bias=use_bias)
    output = conv1(inputs)

    with self.test_session():
      tf.variables_initializer(
          [conv1.w, conv1.b] if use_bias else [conv1.w]).run()
      output_eval = output.eval({inputs: np.zeros(self.input_shape)})
      self.assertEqual(output_eval.shape, tuple(self.output_shape))

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testKernelShape(self, use_bias):
    """Test that errors are thrown for invalid kernel shapes."""

    snt.DepthwiseConv2D(channel_multiplier=1, kernel_shape=[3, 4])
    snt.DepthwiseConv2D(channel_multiplier=1, kernel_shape=3)
    error_msg = (r"Invalid kernel shape: x is \[3], must be either a positive"
                 r" integer or an iterable of positive integers of size 2")
    with self.assertRaisesRegexp(snt.IncompatibleShapeError, error_msg):
      snt.DepthwiseConv2D(channel_multiplier=1, kernel_shape=[3],
                          use_bias=use_bias, name="conv1")

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testStrideError(self, use_bias):
    """Test that errors are thrown for invalid strides."""

    snt.DepthwiseConv2D(channel_multiplier=1, kernel_shape=3, stride=1,
                        use_bias=use_bias)
    snt.DepthwiseConv2D(channel_multiplier=1, kernel_shape=3, stride=[1] * 2,
                        use_bias=use_bias)
    snt.DepthwiseConv2D(channel_multiplier=1, kernel_shape=3, stride=[1] * 4,
                        use_bias=use_bias)

    error_msg = (r"stride is \[1, 1, 1\] \(.*\), must be either a positive "
                 r"integer or an iterable of positive integers of size 2")
    with self.assertRaisesRegexp(snt.IncompatibleShapeError, error_msg):
      snt.DepthwiseConv2D(channel_multiplier=3,
                          kernel_shape=3,
                          stride=[1, 1, 1],
                          use_bias=use_bias,
                          name="conv1")

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testInputTypeError(self, use_bias):
    """Test that errors are thrown for invalid input types."""
    conv1 = snt.DepthwiseConv2D(
        channel_multiplier=3,
        kernel_shape=3,
        stride=1,
        padding=snt.SAME,
        use_bias=use_bias,
        initializers=create_constant_initializers(1.0, 1.0, use_bias))

    for dtype in (tf.float16, tf.float64):
      x = tf.constant(np.ones([1, 5, 5, 1]), dtype=dtype)
      err = "Input must have dtype tf.float32.*"
      with self.assertRaisesRegexp(TypeError, err):
        conv1(x)

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testInitializers(self, use_bias):
    """Test that initializers work as expected."""
    w = random.random()
    b = np.random.randn(6)  # Kernel shape is 3, input channels are 2, 2*3 = 6

    conv1 = snt.DepthwiseConv2D(
        channel_multiplier=3,
        kernel_shape=3,
        stride=1,
        use_bias=use_bias,
        initializers=create_constant_initializers(w, b, use_bias))

    conv1(tf.placeholder(tf.float32, [1, 10, 10, 2]))

    with self.test_session():
      tf.variables_initializer(
          [conv1.w, conv1.b] if use_bias else [conv1.w]).run()

      self.assertAllClose(
          conv1.w.eval(), np.full(
              [3, 3, 2, 3], w, dtype=np.float32))

      if use_bias:
        self.assertAllClose(conv1.b.eval(), b)

    error_msg = "Initializer for 'w' is not a callable function"
    with self.assertRaisesRegexp(TypeError, error_msg):
      snt.DepthwiseConv2D(
          channel_multiplier=3,
          kernel_shape=3,
          stride=1,
          use_bias=use_bias,
          initializers={"w": tf.ones([])})

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testRegularizersInRegularizationLosses(self, use_bias):
    regularizers = create_regularizers(
        use_bias, tf.contrib.layers.l1_regularizer(scale=0.5))

    conv1 = snt.DepthwiseConv2D(
        channel_multiplier=3,
        kernel_shape=3,
        stride=1,
        regularizers=regularizers,
        use_bias=use_bias,
        name="conv1")
    conv1(tf.placeholder(tf.float32, [1, 10, 10, 2]))

    graph_regularizers = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    self.assertRegexpMatches(graph_regularizers[0].name, ".*l1_regularizer.*")
    if use_bias:
      self.assertRegexpMatches(graph_regularizers[1].name, ".*l1_regularizer.*")

  def testInitializerMutation(self):
    """Test that initializers are not mutated."""

    initializers = {"b": tf.constant_initializer(0)}
    initializers_copy = dict(initializers)

    conv1 = snt.DepthwiseConv2D(
        channel_multiplier=3,
        kernel_shape=3,
        stride=1,
        initializers=initializers)

    conv1(tf.placeholder(tf.float32, [10, 10, 1, 2]))

    self.assertAllEqual(initializers, initializers_copy)

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testComputationSame(self, use_bias):
    """Run through for something with a known answer using SAME padding."""
    conv1 = snt.DepthwiseConv2D(
        channel_multiplier=1,
        kernel_shape=[3, 3],
        stride=1,
        padding=snt.SAME,
        use_bias=use_bias,
        initializers=create_constant_initializers(1.0, 1.0, use_bias))

    out = conv1(tf.constant(np.ones([1, 5, 5, 1], dtype=np.float32)))
    expected_out = np.array([[5, 7, 7, 7, 5],
                             [7, 10, 10, 10, 7],
                             [7, 10, 10, 10, 7],
                             [7, 10, 10, 10, 7],
                             [5, 7, 7, 7, 5]])
    if not use_bias:
      expected_out -= 1

    with self.test_session():
      tf.variables_initializer(
          [conv1.w, conv1.b] if use_bias else [conv1.w]).run()

      self.assertAllClose(np.reshape(out.eval(), [5, 5]), expected_out)

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testComputationValid(self, use_bias):
    """Run through for something with a known answer using snt.VALID padding."""
    conv1 = snt.DepthwiseConv2D(
        channel_multiplier=1,
        kernel_shape=[3, 3],
        stride=1,
        padding=snt.VALID,
        use_bias=use_bias,
        initializers=create_constant_initializers(1.0, 1.0, use_bias))

    out = conv1(tf.constant(np.ones([1, 5, 5, 1], dtype=np.float32)))
    expected_out = np.array([[10, 10, 10],
                             [10, 10, 10],
                             [10, 10, 10]])
    if not use_bias:
      expected_out -= 1

    with self.test_session():
      tf.variables_initializer(
          [conv1.w, conv1.b] if use_bias else [conv1.w]).run()

      self.assertAllClose(np.reshape(out.eval(), [3, 3]), expected_out)

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testComputationValidMultiChannel(self, use_bias):
    """Run through for something with a known answer using snt.VALID padding."""
    conv1 = snt.DepthwiseConv2D(
        channel_multiplier=1,
        kernel_shape=[3, 3],
        stride=1,
        padding=snt.VALID,
        use_bias=use_bias,
        initializers=create_constant_initializers(1.0, 1.0, use_bias))

    out = conv1(tf.constant(np.ones([1, 5, 5, 3], dtype=np.float32)))
    expected_out = np.array([[[10] * 3] * 3] * 3)
    if not use_bias:
      expected_out -= 1

    with self.test_session():
      tf.variables_initializer(
          [conv1.w, conv1.b] if use_bias else [conv1.w]).run()

      self.assertAllClose(
          np.reshape(out.eval(), [3, 3, 3]), expected_out)

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testSharing(self, use_bias):
    """Sharing is working."""
    conv1 = snt.DepthwiseConv2D(
        channel_multiplier=3, kernel_shape=3, stride=1, padding=snt.SAME,
        use_bias=use_bias)

    x = np.random.randn(1, 5, 5, 1)
    x1 = tf.constant(x, dtype=np.float32)
    x2 = tf.constant(x, dtype=np.float32)

    out1 = conv1(x1)
    out2 = conv1(x2)

    with self.test_session():
      tf.variables_initializer(
          [conv1.w, conv1.b] if use_bias else [conv1.w]).run()
      self.assertAllClose(out1.eval(), out2.eval())

      # Kernel shape was set to 3, which is expandeded to [3, 3, 3].
      # Input channels are 1, output channels := in_channels * multiplier.
      # multiplier is kernel_shape[2] == 3. So weight layout must be:
      # (3, 3, 1, 3).
      w = np.random.randn(3, 3, 1, 3)  # Now change the weights.
      conv1.w.assign(w).eval()
      self.assertAllClose(out1.eval(), out2.eval())


class SeparableConv2DTest(parameterized.ParameterizedTestCase,
                          tf.test.TestCase):

  def setUp(self):
    """Set up some variables to re-use in multiple tests."""

    super(SeparableConv2DTest, self).setUp()

    self.batch_size = batch_size = random.randint(1, 100)
    self.in_height = in_height = random.randint(10, 188)
    self.in_width = in_width = random.randint(10, 188)
    self.in_channels = in_channels = random.randint(1, 10)
    self.input_shape = [batch_size, in_height, in_width, in_channels]

    self.kernel_shape_h = kernel_shape_h = random.randint(1, 10)
    self.kernel_shape_w = kernel_shape_w = random.randint(1, 10)
    self.channel_multiplier = channel_multiplier = random.randint(1, 10)
    self.kernel_shape = [kernel_shape_h, kernel_shape_w]

    self.out_channels_dw = out_channels_dw = in_channels * channel_multiplier
    self.output_shape = [batch_size, in_height, in_width, out_channels_dw]
    self.depthwise_filter_shape = [
        kernel_shape_h, kernel_shape_w, in_channels, channel_multiplier
    ]
    self.pointwise_filter_shape = [1, 1, out_channels_dw, out_channels_dw]

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testShapesSame(self, use_bias):
    """Test that the generated shapes are correct with SAME padding."""

    out_channels = self.out_channels_dw
    input_shape = self.input_shape
    kernel_shape = self.kernel_shape
    output_shape = self.output_shape
    depthwise_filter_shape = self.depthwise_filter_shape
    pointwise_filter_shape = self.pointwise_filter_shape
    channel_multiplier = self.channel_multiplier

    inputs = tf.placeholder(tf.float32, shape=input_shape)

    conv1 = snt.SeparableConv2D(
        output_channels=out_channels,
        channel_multiplier=channel_multiplier,
        kernel_shape=kernel_shape,
        padding=snt.SAME,
        use_bias=use_bias)

    output = conv1(inputs)

    self.assertTrue(output.get_shape().is_compatible_with(output_shape))
    self.assertTrue(conv1.w_dw.get_shape().is_compatible_with(
        depthwise_filter_shape))
    self.assertTrue(conv1.w_pw.get_shape().is_compatible_with(
        pointwise_filter_shape))
    if use_bias:
      self.assertTrue(conv1.b.get_shape().is_compatible_with([out_channels]))

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testShapesNotKnown(self, use_bias):
    """Test that the generated shapes are correct when input shape not known."""

    inputs = tf.placeholder(
        tf.float32, shape=[None, None, None, self.in_channels], name="inputs")

    conv1 = snt.SeparableConv2D(
        output_channels=self.out_channels_dw,
        channel_multiplier=1,
        kernel_shape=self.kernel_shape,
        padding=snt.SAME,
        use_bias=use_bias)
    output = conv1(inputs)

    with self.test_session():
      tf.variables_initializer(
          [conv1.w_dw, conv1.w_pw, conv1.b] if use_bias else
          [conv1.w_dw, conv1.w_pw]).run()
      output_eval = output.eval({inputs: np.zeros(self.input_shape)})
      self.assertEqual(output_eval.shape, tuple(self.output_shape))

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testKernelShape(self, use_bias):
    """Test that errors are thrown for invalid kernel shapes."""

    # No check against output_channels is done yet (needs input size).
    snt.SeparableConv2D(
        output_channels=1,
        channel_multiplier=2,
        kernel_shape=[3, 4],
        name="conv1",
        use_bias=use_bias)
    snt.SeparableConv2D(
        output_channels=1, channel_multiplier=1, kernel_shape=3, name="conv1")

    error_msg = (r"Invalid kernel shape: x is \[3], must be either a positive"
                 r" integer or an iterable of positive integers of size 2")
    with self.assertRaisesRegexp(snt.IncompatibleShapeError, error_msg):
      snt.SeparableConv2D(output_channels=1,
                          channel_multiplier=3,
                          kernel_shape=[3],
                          use_bias=use_bias)

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testStrideError(self, use_bias):
    """Test that errors are thrown for invalid strides."""

    snt.SeparableConv2D(
        output_channels=1, channel_multiplier=3, kernel_shape=3, stride=1,
        use_bias=use_bias)
    snt.SeparableConv2D(
        output_channels=1, channel_multiplier=3, kernel_shape=3, stride=[1, 1],
        use_bias=use_bias)
    snt.SeparableConv2D(
        output_channels=1,
        channel_multiplier=3,
        kernel_shape=3,
        stride=[1, 1, 1, 1],
        use_bias=use_bias)

    error_msg = (r"stride is \[1, 1, 1\] \(.*\), must be either a positive "
                 r"integer or an iterable of positive integers of size 2")
    with self.assertRaisesRegexp(snt.IncompatibleShapeError, error_msg):
      snt.SeparableConv2D(output_channels=1,
                          channel_multiplier=3,
                          kernel_shape=3,
                          stride=[1, 1, 1],
                          name="conv1",
                          use_bias=use_bias)

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testInputTypeError(self, use_bias):
    """Test that errors are thrown for invalid input types."""
    conv1 = snt.SeparableConv2D(
        output_channels=3,
        channel_multiplier=1,
        kernel_shape=3,
        padding=snt.SAME,
        use_bias=use_bias,
        initializers=create_separable_constant_initializers(
            1.0, 1.0, 1.0, use_bias))

    for dtype in (tf.float16, tf.float64):
      x = tf.constant(np.ones([1, 5, 5, 1]), dtype=dtype)
      err = "Input must have dtype tf.float32.*"
      with self.assertRaisesRegexp(TypeError, err):
        conv1(x)

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testInitializers(self, use_bias):
    """Test that initializers work as expected."""

    w_dw = random.random()
    w_pw = random.random()
    b = np.random.randn(6)  # Kernel shape is 3, input channels are 2, 2*3 = 6.
    conv1 = snt.SeparableConv2D(
        output_channels=6,
        channel_multiplier=3,
        kernel_shape=3,
        use_bias=use_bias,
        initializers=create_separable_constant_initializers(
            w_dw, w_pw, b, use_bias))

    conv1(tf.placeholder(tf.float32, [1, 10, 10, 2]))

    with self.test_session():
      tf.variables_initializer(
          [conv1.w_dw, conv1.w_pw, conv1.b] if use_bias else
          [conv1.w_dw, conv1.w_pw]).run()

      self.assertAllClose(
          conv1.w_dw.eval(), np.full(
              [3, 3, 2, 3], w_dw, dtype=np.float32))
      self.assertAllClose(
          conv1.w_pw.eval(), np.full(
              [1, 1, 6, 6], w_pw, dtype=np.float32))

      if use_bias:
        self.assertAllClose(conv1.b.eval(), b)

    error_msg = "Initializer for 'w_dw' is not a callable function"
    with self.assertRaisesRegexp(TypeError, error_msg):
      snt.SeparableConv2D(
          output_channels=3,
          channel_multiplier=1,
          kernel_shape=3,
          stride=1,
          use_bias=use_bias,
          initializers={"w_dw": tf.ones([])})

  def testInitializerMutation(self):
    """Test that initializers are not mutated."""

    initializers = {"b": tf.constant_initializer(0)}
    initializers_copy = dict(initializers)

    conv1 = snt.SeparableConv2D(
        output_channels=3,
        channel_multiplier=1,
        kernel_shape=3,
        stride=1,
        initializers=initializers)

    conv1(tf.placeholder(tf.float32, [10, 10, 1, 2]))

    self.assertAllEqual(initializers, initializers_copy)

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testRegularizersInRegularizationLosses(self, use_bias):
    regularizers = create_separable_regularizers(
        use_bias, tf.contrib.layers.l1_regularizer(scale=0.5))

    conv1 = snt.SeparableConv2D(
        output_channels=3,
        channel_multiplier=1,
        kernel_shape=3,
        stride=1,
        regularizers=regularizers,
        use_bias=use_bias,
        name="conv1")
    conv1(tf.placeholder(tf.float32, [10, 10, 1, 2]))

    graph_regularizers = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    self.assertRegexpMatches(graph_regularizers[0].name, ".*l1_regularizer.*")
    self.assertRegexpMatches(graph_regularizers[1].name, ".*l1_regularizer.*")
    if use_bias:
      self.assertRegexpMatches(graph_regularizers[2].name, ".*l1_regularizer.*")

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testComputationSame(self, use_bias):
    """Run through for something with a known answer using SAME padding."""

    conv1 = snt.SeparableConv2D(
        output_channels=1,
        channel_multiplier=1,
        kernel_shape=[3, 3],
        padding=snt.SAME,
        name="conv1",
        use_bias=use_bias,
        initializers=create_separable_constant_initializers(
            1.0, 1.0, 1.0, use_bias))

    out = conv1(tf.constant(np.ones([1, 5, 5, 1], dtype=np.float32)))
    expected_out = np.array([[5, 7, 7, 7, 5],
                             [7, 10, 10, 10, 7],
                             [7, 10, 10, 10, 7],
                             [7, 10, 10, 10, 7],
                             [5, 7, 7, 7, 5]])
    if not use_bias:
      expected_out -= 1

    with self.test_session():
      tf.variables_initializer(
          [conv1.w_dw, conv1.w_pw, conv1.b] if use_bias else
          [conv1.w_dw, conv1.w_pw]).run()

      self.assertAllClose(np.reshape(out.eval(), [5, 5]), expected_out)

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testComputationValid(self, use_bias):
    """Run through for something with a known answer using snt.VALID padding."""

    conv1 = snt.SeparableConv2D(
        output_channels=1,
        channel_multiplier=1,
        kernel_shape=[3, 3],
        padding=snt.VALID,
        use_bias=use_bias,
        initializers=create_separable_constant_initializers(
            1.0, 1.0, 1.0, use_bias))

    out = conv1(tf.constant(np.ones([1, 5, 5, 1], dtype=np.float32)))
    expected_out = np.array([[10, 10, 10],
                             [10, 10, 10],
                             [10, 10, 10]])
    if not use_bias:
      expected_out -= 1

    with self.test_session():
      tf.variables_initializer(
          [conv1.w_dw, conv1.w_pw, conv1.b] if use_bias else
          [conv1.w_dw, conv1.w_pw]).run()

      self.assertAllClose(np.reshape(out.eval(), [3, 3]), expected_out)

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testComputationValidMultiChannel(self, use_bias):
    """Run through for something with a known answer using snt.VALID padding."""

    conv1 = snt.SeparableConv2D(
        output_channels=3,
        channel_multiplier=1,
        kernel_shape=[3, 3],
        padding=snt.VALID,
        use_bias=use_bias,
        initializers=create_separable_constant_initializers(
            1.0, 1.0, 1.0, use_bias))

    out = conv1(tf.constant(np.ones([1, 5, 5, 3], dtype=np.float32)))
    expected_out = np.array([[[28] * 3] * 3] * 3)
    if not use_bias:
      expected_out -= 1

    with self.test_session():
      tf.variables_initializer(
          [conv1.w_dw, conv1.w_pw, conv1.b] if use_bias else
          [conv1.w_dw, conv1.w_pw]).run()

      self.assertAllClose(np.reshape(out.eval(), [3, 3, 3]), expected_out)

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testComputationValidChannelMultiplier(self, use_bias):
    """Run through for something with a known answer using snt.VALID padding."""

    input_channels = 3
    channel_multiplier = 5
    output_channels = input_channels * channel_multiplier
    conv1 = snt.SeparableConv2D(
        output_channels=output_channels,
        channel_multiplier=channel_multiplier,
        kernel_shape=[3, 3],
        padding=snt.VALID,
        use_bias=use_bias,
        initializers=create_separable_constant_initializers(
            1.0, 1.0, 1.0, use_bias))

    input_data = np.ones([1, 5, 5, input_channels], dtype=np.float32)
    out = conv1(tf.constant(input_data))
    expected_out = np.ones((3, 3, output_channels)) * 136
    if not use_bias:
      expected_out -= 1

    self.assertTrue(out.get_shape().is_compatible_with([1, 3, 3, output_channels
                                                       ]))

    with self.test_session():
      tf.variables_initializer(
          [conv1.w_dw, conv1.w_pw, conv1.b] if use_bias else
          [conv1.w_dw, conv1.w_pw]).run()

      self.assertAllClose(np.reshape(out.eval(), [3, 3, output_channels]),
                          expected_out)
      # Each convolution with weight 1 and size 3x3 results in an output of 9.
      # Pointwise filter is [1, 1, input_channels * channel_multiplier = 15, x].
      # Results in 9 * 15 = 135 + 1 bias = 136 as outputs.

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testSharing(self, use_bias):
    """Sharing is working."""
    conv1 = snt.SeparableConv2D(
        output_channels=3, channel_multiplier=3, kernel_shape=3,
        use_bias=use_bias)

    x = np.random.randn(1, 5, 5, 1)
    x1 = tf.constant(x, dtype=np.float32)
    x2 = tf.constant(x, dtype=np.float32)

    out1 = conv1(x1)
    out2 = conv1(x2)

    with self.test_session():
      tf.variables_initializer(
          [conv1.w_dw, conv1.w_pw, conv1.b] if use_bias else
          [conv1.w_dw, conv1.w_pw]).run()
      self.assertAllClose(out1.eval(), out2.eval())

      # Kernel shape was set to 3, which is expandeded to [3, 3, 3].
      # Input channels are 1, output channels := in_channels * multiplier.
      # multiplier is kernel_shape[2] == 3. So weight layout must be:
      # (3, 3, 1, 3).
      w_dw = np.random.randn(3, 3, 1, 3)  # Now change the weights.
      w_pw = np.random.randn(1, 1, 3, 3)  # Now change the weights.
      conv1.w_dw.assign(w_dw).eval()
      conv1.w_pw.assign(w_pw).eval()
      self.assertAllClose(out1.eval(), out2.eval())


class Conv3DTest(parameterized.ParameterizedTestCase, tf.test.TestCase):

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testShapesSame(self, use_bias):
    """The generated shapes are correct with SAME padding."""

    batch_size = random.randint(1, 100)
    in_depth = random.randint(10, 288)
    in_height = random.randint(10, 288)
    in_width = random.randint(10, 288)
    in_channels = random.randint(1, 10)
    out_channels = random.randint(1, 32)
    kernel_shape_d = random.randint(1, 11)
    kernel_shape_h = random.randint(1, 11)
    kernel_shape_w = random.randint(1, 11)

    inputs = tf.placeholder(
        tf.float32,
        shape=[batch_size, in_depth, in_height, in_width, in_channels])

    conv1 = snt.Conv3D(
        output_channels=out_channels,
        kernel_shape=[kernel_shape_d, kernel_shape_h, kernel_shape_w],
        padding=snt.SAME,
        stride=1,
        use_bias=use_bias,
        name="conv1")

    output = conv1(inputs)

    self.assertTrue(
        output.get_shape().is_compatible_with(
            [batch_size, in_depth, in_height, in_width, out_channels]))

    self.assertTrue(
        conv1.w.get_shape().is_compatible_with(
            [kernel_shape_d, kernel_shape_h, kernel_shape_w, in_channels,
             out_channels]))
    if use_bias:
      self.assertTrue(
          conv1.b.get_shape().is_compatible_with(
              [out_channels]))

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testShapesWithUnknownInputShape(self, use_bias):
    """The generated shapes are correct when input shape not known."""

    batch_size = 5
    in_depth = in_height = in_width = 32
    in_channels = out_channels = 5
    kernel_shape_d = kernel_shape_h = kernel_shape_w = 3

    inputs = tf.placeholder(
        tf.float32,
        shape=[None, None, None, None, in_channels],
        name="inputs")

    conv1 = snt.Conv3D(
        name="conv1",
        output_channels=out_channels,
        kernel_shape=[kernel_shape_d, kernel_shape_h, kernel_shape_w],
        padding=snt.SAME,
        stride=1,
        use_bias=use_bias)

    output = conv1(inputs)

    with self.test_session():
      tf.variables_initializer(
          [conv1.w, conv1.b] if use_bias else [conv1.w]).run()

      output_eval = output.eval({
          inputs: np.zeros([batch_size, in_depth, in_height, in_width,
                            in_channels])})

      self.assertEqual(
          output_eval.shape,
          (batch_size, in_depth, in_height, in_width, out_channels))

  def testKernelShape(self):
    """Errors are thrown for invalid kernel shapes."""

    snt.Conv3D(output_channels=10, kernel_shape=[3, 4, 5], name="conv1")
    snt.Conv3D(output_channels=10, kernel_shape=3, name="conv1")

    with self.assertRaisesRegexp(snt.Error, "Invalid kernel shape.*"):
      snt.Conv3D(output_channels=10, kernel_shape=[3, 3], name="conv1")
      snt.Conv3D(output_channels=10, kernel_shape=[3, 3, 3, 3], name="conv1")

  def testStrideError(self):
    """Errors are thrown for invalid strides."""

    snt.Conv3D(
        output_channels=10, kernel_shape=3, stride=1, name="conv1")
    snt.Conv3D(
        output_channels=10, kernel_shape=3, stride=[1, 1, 1], name="conv1")
    snt.Conv3D(
        output_channels=10, kernel_shape=3, stride=[1, 1, 1, 1, 1],
        name="conv1")

    with self.assertRaisesRegexp(snt.Error, "Invalid stride.*"):
      snt.Conv3D(output_channels=10, kernel_shape=3, stride=[1, 1],
                 name="conv1")
      snt.Conv3D(output_channels=10, kernel_shape=3, stride=[1, 1, 1, 1],
                 name="conv1")

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testRateError(self, use_bias):
    """Errors are thrown for invalid dilation rates."""

    snt.Conv3D(
        output_channels=10, kernel_shape=3, rate=1, name="conv1",
        use_bias=use_bias)
    snt.Conv3D(
        output_channels=10, kernel_shape=3, rate=2, name="conv1",
        use_bias=use_bias)

    for rate in [0, 0.5, -1]:
      with self.assertRaisesRegexp(snt.IncompatibleShapeError,
                                   "Invalid rate shape*"):
        snt.Conv3D(output_channels=10,
                   kernel_shape=3,
                   rate=rate,
                   name="conv1")

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testRateAndStrideError(self, use_bias):
    """Errors are thrown for stride > 1 when using atrous convolution."""
    err = "Cannot have stride > 1 with rate > 1"
    with self.assertRaisesRegexp(snt.NotSupportedError, err):
      snt.Conv3D(output_channels=10, kernel_shape=3,
                 stride=2, rate=2, name="conv1", use_bias=use_bias)
    with self.assertRaisesRegexp(snt.NotSupportedError, err):
      snt.Conv3D(output_channels=10, kernel_shape=3,
                 stride=[2, 2, 1], rate=2, name="conv1", use_bias=use_bias)

  def testInputTypeError(self):
    """Errors are thrown for invalid input types."""
    conv1 = snt.Conv3D(output_channels=1,
                       kernel_shape=3,
                       stride=1,
                       padding=snt.SAME,
                       name="conv1",
                       initializers={
                           "w": tf.constant_initializer(1.0),
                           "b": tf.constant_initializer(1.0),
                       })

    for dtype in (tf.float16, tf.float64):
      x = tf.constant(np.ones([1, 5, 5, 5, 1]), dtype=dtype)
      self.assertRaisesRegexp(TypeError, "Input must have dtype tf.float32.*",
                              conv1, x)

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testInitializers(self, use_bias):
    """Test initializers work as expected."""

    w = random.random()
    b = random.random()
    conv1 = snt.Conv3D(
        output_channels=1,
        kernel_shape=3,
        stride=1,
        name="conv1",
        use_bias=use_bias,
        initializers=create_constant_initializers(w, b, use_bias))

    conv1(tf.placeholder(tf.float32, [1, 10, 10, 10, 2]))

    with self.test_session():
      tf.variables_initializer(
          [conv1.w, conv1.b] if use_bias else [conv1.w]).run()

      self.assertAllClose(
          conv1.w.eval(),
          np.full([3, 3, 3, 2, 1], w, dtype=np.float32))

      if use_bias:
        self.assertAllClose(
            conv1.b.eval(),
            [b])

    with self.assertRaises(TypeError):
      snt.Conv3D(output_channels=10, kernel_shape=3, stride=1, name="conv1",
                 initializers={"w": tf.ones([])})

  def testInitializerMutation(self):
    """Test that initializers are not mutated."""

    initializers = {"b": tf.constant_initializer(0)}
    initializers_copy = dict(initializers)

    conv1 = snt.Conv3D(
        output_channels=1,
        kernel_shape=3,
        stride=1,
        name="conv1",
        initializers=initializers)

    conv1(tf.placeholder(tf.float32, [1, 10, 10, 10, 2]))

    self.assertAllEqual(initializers, initializers_copy)

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testRegularizersInRegularizationLosses(self, use_bias):
    regularizers = create_regularizers(
        use_bias, tf.contrib.layers.l1_regularizer(scale=0.5))

    conv1 = snt.Conv3D(
        output_channels=1,
        kernel_shape=3,
        stride=1,
        regularizers=regularizers,
        use_bias=use_bias,
        name="conv1")
    conv1(tf.placeholder(tf.float32, [1, 10, 10, 10, 2]))

    graph_regularizers = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    self.assertRegexpMatches(graph_regularizers[0].name, ".*l1_regularizer.*")
    if use_bias:
      self.assertRegexpMatches(graph_regularizers[1].name, ".*l1_regularizer.*")

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testComputationSame(self, use_bias):
    """Run through for something with a known answer using SAME padding."""

    conv1 = snt.Conv3D(
        output_channels=1,
        kernel_shape=3,
        stride=1,
        padding=snt.SAME,
        name="conv1",
        use_bias=use_bias,
        initializers=create_constant_initializers(1.0, 1.0, use_bias))

    out = conv1(tf.constant(np.ones([1, 5, 5, 5, 1], dtype=np.float32)))
    expected_out = np.asarray([9, 13, 13, 13, 9, 13, 19, 19, 19, 13, 13, 19, 19,
                               19, 13, 13, 19, 19, 19, 13, 9, 13, 13, 13, 9, 13,
                               19, 19, 19, 13, 19, 28, 28, 28, 19, 19, 28, 28,
                               28, 19, 19, 28, 28, 28, 19, 13, 19, 19, 19, 13,
                               13, 19, 19, 19, 13, 19, 28, 28, 28, 19, 19, 28,
                               28, 28, 19, 19, 28, 28, 28, 19, 13, 19, 19, 19,
                               13, 13, 19, 19, 19, 13, 19, 28, 28, 28, 19, 19,
                               28, 28, 28, 19, 19, 28, 28, 28, 19, 13, 19, 19,
                               19, 13, 9, 13, 13, 13, 9, 13, 19, 19, 19, 13, 13,
                               19, 19, 19, 13, 13, 19, 19, 19, 13, 9, 13, 13,
                               13, 9]).reshape((5, 5, 5))

    if not use_bias:
      expected_out -= 1

    with self.test_session():
      tf.variables_initializer(
          [conv1.w, conv1.b] if use_bias else [conv1.w]).run()

      self.assertAllClose(
          np.reshape(out.eval(), [5, 5, 5]), expected_out)

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testComputationValid(self, use_bias):
    """Run through for something with a known answer using snt.VALID padding."""

    conv1 = snt.Conv3D(
        output_channels=1,
        kernel_shape=3,
        stride=1,
        padding=snt.VALID,
        name="conv1",
        use_bias=use_bias,
        initializers=create_constant_initializers(1.0, 1.0, use_bias))

    out = conv1(tf.constant(np.ones([1, 5, 5, 5, 1], dtype=np.float32)))
    expected_out = np.asarray([28] * 27).reshape((3, 3, 3))

    if not use_bias:
      expected_out -= 1

    with self.test_session():
      tf.variables_initializer(
          [conv1.w, conv1.b] if use_bias else [conv1.w]).run()

      self.assertAllClose(
          np.reshape(out.eval(), [3, 3, 3]), expected_out)

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testSharing(self, use_bias):
    """Sharing is working."""

    conv1 = snt.Conv3D(
        output_channels=1,
        kernel_shape=3,
        stride=1,
        padding=snt.SAME,
        use_bias=use_bias,
        name="conv1")

    x = np.random.randn(1, 5, 5, 5, 1)
    x1 = tf.constant(x, dtype=np.float32)
    x2 = tf.constant(x, dtype=np.float32)

    out1 = conv1(x1)
    out2 = conv1(x2)

    with self.test_session():
      tf.variables_initializer(
          [conv1.w, conv1.b] if use_bias else [conv1.w]).run()

      self.assertAllClose(
          out1.eval(),
          out2.eval())

      # Now change the weights
      w = np.random.randn(3, 3, 3, 1, 1)
      conv1.w.assign(w).eval()

      self.assertAllClose(
          out1.eval(),
          out2.eval())


class Conv3DTransposeTest(parameterized.ParameterizedTestCase,
                          tf.test.TestCase):

  def setUp(self):
    """Set up some variables to re-use in multiple tests."""

    super(Conv3DTransposeTest, self).setUp()

    self.batch_size = 7
    self.in_depth = 7
    self.in_height = 7
    self.in_width = 11
    self.in_channels = 4
    self.out_channels = 10
    self.kernel_shape_d = 5
    self.kernel_shape_h = 5
    self.kernel_shape_w = 7
    self.stride_d = 1
    self.stride_h = 1
    self.stride_w = 1
    self.padding = snt.SAME

    self.in_shape = (self.batch_size, self.in_depth, self.in_height,
                     self.in_width, self.in_channels)

    self.out_shape = (self.in_depth, self.in_height, self.in_width)

    self.kernel_shape = (self.kernel_shape_d, self.kernel_shape_h,
                         self.kernel_shape_w)

    self.kernel_shape2 = (self.kernel_shape_d, self.kernel_shape_h,
                          self.kernel_shape_w, self.out_channels,
                          self.in_channels)

    self.strides = (self.stride_d, self.stride_h, self.stride_w)

  def testKernelsNotSpecified(self):
    with self.assertRaisesRegexp(ValueError, "`kernel_shape` cannot be None."):
      snt.Conv3DTranspose(output_channels=1)

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testOutputShapeConsistency(self, use_bias):
    """Tests if output shapes are valid."""

    # When padding is SAME, then the actual number of padding pixels can be
    # computed as: pad = kernel_shape - strides + (-input_shape % strides)
    #                 =     5         -    1    + (- 32       %      1) = 4

    # The formula for the minimal size is:
    # oH = strides[1] * (in_height - 1) - padding + kernel_shape_h
    # oH =          1 * (       32 - 1) -    4    +       5 = 32

    # The formula for the maximum size (due to extra pixels) is:
    # oH_max = oH + strides[1] - 1
    # so, for strides = 1 and padding = SAME, input size == output size.
    inputs = tf.placeholder(tf.float32, shape=self.in_shape)

    conv1 = snt.Conv3DTranspose(name="conv3d_1",
                                output_channels=self.out_channels,
                                output_shape=self.out_shape,
                                kernel_shape=self.kernel_shape,
                                padding=self.padding,
                                stride=1,
                                use_bias=use_bias)

    outputs = conv1(inputs)

    self.assertTrue(outputs.get_shape().is_compatible_with((
        self.batch_size,) + self.out_shape + (self.out_channels,)))

    self.assertTrue(conv1.w.get_shape().is_compatible_with(self.kernel_shape2))
    if use_bias:
      self.assertTrue(conv1.b.get_shape().is_compatible_with(
          [self.out_channels]))

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testOutputShapeInteger(self, use_bias):
    """Tests if output shapes are valid when specified as an integer."""
    inputs = tf.zeros(shape=[3, 5, 5, 5, 2], dtype=tf.float32)
    inputs_2 = tf.zeros(shape=[3, 5, 7, 5, 2], dtype=tf.float32)

    conv1 = snt.Conv3DTranspose(name="conv3d_1",
                                output_channels=10,
                                output_shape=10,
                                kernel_shape=5,
                                padding=snt.SAME,
                                stride=2,
                                use_bias=use_bias)

    outputs = conv1(inputs)
    outputs_2 = conv1(inputs_2)

    self.assertTrue(outputs.get_shape().is_compatible_with((3, 10, 10, 10, 10)))

    with self.test_session() as sess:
      tf.global_variables_initializer().run()
      sess.run(outputs)
      with self.assertRaises(tf.errors.InvalidArgumentError):
        sess.run(outputs_2)

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testTransposition(self, use_bias):
    """Tests if the correct ouput shapes are setup in transposed module."""
    net = snt.Conv3DTranspose(name="conv3d_3",
                              output_channels=self.out_channels,
                              output_shape=self.out_shape,
                              kernel_shape=self.kernel_shape,
                              padding=self.padding,
                              stride=1,
                              use_bias=use_bias)

    net_transpose = net.transpose()
    input_to_net = tf.placeholder(tf.float32, shape=self.in_shape)
    err = "Variables in {} not instantiated yet, __call__ the module first."
    with self.assertRaisesRegexp(snt.NotConnectedError,
                                 err.format(net.scope_name)):
      net_transpose(input_to_net)
    net_transpose = net.transpose(name="another_net_transpose")
    net_out = net(input_to_net)
    net_transposed_output = net_transpose(net_out)
    self.assertEqual(net_transposed_output.get_shape(),
                     input_to_net.get_shape())


if __name__ == "__main__":
  tf.test.main()
