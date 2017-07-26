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

"""Tests for sonnet.python.modules.basic."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

# Dependency imports

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import sonnet as snt
from sonnet.testing import parameterized
import tensorflow as tf

from tensorflow.python.client import device_lib
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variables
from tensorflow.python.util import nest


def _test_initializer(mu=0.0, sigma=1.0, dtype=tf.float32):
  """Custom initializer for Linear tests."""
  def _initializer(shape,
                   dtype=init_ops._assert_float_dtype(dtype),
                   partition_info=None):  # pylint: disable=unused-argument
    random_normal_tensor = np.asarray(np.random.randn(*shape)) * sigma + mu
    return random_normal_tensor.astype(dtype.as_numpy_dtype())
  return _initializer


class LinearTest(tf.test.TestCase, parameterized.ParameterizedTestCase):

  def setUp(self):
    super(LinearTest, self).setUp()

    self.batch_size = 11
    self.in_size = 13
    self.out_size = 17
    self.seed = 42

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testShape(self, use_bias):
    inputs = tf.placeholder(tf.float32, shape=[self.batch_size, self.in_size])
    lin = snt.Linear(output_size=self.out_size, use_bias=use_bias)
    output = lin(inputs)
    self.assertTrue(
        output.get_shape().is_compatible_with([self.batch_size, self.out_size]))

  def testName(self):
    mod_name = "unique_name"
    with tf.variable_scope("scope"):
      lin = snt.Linear(name=mod_name, output_size=self.out_size)
    self.assertEqual(lin.scope_name, "scope/" + mod_name)
    self.assertEqual(lin.module_name, mod_name)

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testVariables(self, use_bias):
    inputs = tf.placeholder(tf.float32, shape=[self.batch_size, self.in_size])
    lin = snt.Linear(output_size=self.out_size, use_bias=use_bias)

    err = r"Variables in {} not instantiated yet, __call__ the module first."
    with self.assertRaisesRegexp(snt.NotConnectedError,
                                 err.format(lin.scope_name)):
      lin.get_variables()

    err = "Variables in {} not instantiated yet, __call__ the module first."
    with self.assertRaisesRegexp(snt.NotConnectedError,
                                 err.format(lin.scope_name)):
      _ = lin.w

    err = "Variables in {} not instantiated yet, __call__ the module first."
    with self.assertRaisesRegexp(snt.NotConnectedError,
                                 err.format(lin.scope_name)):
      _ = lin.b

    lin(inputs)  # Connect the module, but ignore the return value.

    variables_ = lin.get_variables()
    if use_bias:
      self.assertEqual(len(variables_), 2, "Linear should have 2 variables.")
    else:
      err = "No bias Variable in Linear Module when `use_bias=False`."
      with self.assertRaisesRegexp(AttributeError, err):
        _ = lin.b
      self.assertEqual(len(variables_), 1, "Linear should have 1 variable.")

    for v in variables_:
      self.assertRegexpMatches(v.name,
                               r"{}/[wb]:0".format(lin.scope_name))
      if v.name.endswith("w:0"):
        shape = np.ndarray((self.in_size, self.out_size))
      else:
        shape = np.ndarray(self.out_size)
      self.assertShapeEqual(shape, v.initial_value)

  def testCustomGetter(self):
    """Check that custom getters work appropriately."""

    def custom_getter(getter, *args, **kwargs):
      kwargs["trainable"] = False
      return getter(*args, **kwargs)

    inputs = tf.placeholder(tf.float32, shape=[self.batch_size, self.in_size])

    # Make w and b non-trainable.
    lin1 = snt.Linear(output_size=self.out_size,
                      custom_getter=custom_getter)
    lin1(inputs)
    self.assertEqual(0, len(tf.trainable_variables()))
    self.assertEqual(2, len(tf.global_variables()))

    # Make w non-trainable.
    lin2 = snt.Linear(output_size=self.out_size,
                      custom_getter={"w": custom_getter})
    lin2(inputs)
    self.assertEqual(1, len(tf.trainable_variables()))
    self.assertEqual(4, len(tf.global_variables()))

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testComputation(self, use_bias):
    np.random.seed(self.seed)
    types = (tf.float16, tf.float32, tf.float64)
    tol = (1e-2, 1e-6, 1e-9)
    tolerance_map = dict(zip(types, tol))

    for dtype in types:
      inputs = tf.placeholder(dtype, shape=[self.batch_size, self.in_size])

      if use_bias:
        initializers = {"w": _test_initializer(), "b": _test_initializer()}
      else:
        initializers = {"w": _test_initializer()}

      lin = snt.Linear(output_size=self.out_size,
                       use_bias=use_bias,
                       initializers=initializers)
      output = lin(inputs)
      with self.test_session() as sess:
        # With random data, check the TF calculation matches the Numpy version.
        input_data = np.random.randn(self.batch_size,
                                     self.in_size).astype(dtype.as_numpy_dtype)
        sess.run(tf.global_variables_initializer())
        if use_bias:
          output_data, w, b = sess.run([output, lin.w, lin.b],
                                       {inputs: input_data})
        else:
          output_data, w = sess.run([output, lin.w],
                                    {inputs: input_data})

      if use_bias:
        result = (np.dot(input_data, w.astype(dtype.as_numpy_dtype)) +
                  b.astype(dtype.as_numpy_dtype))
      else:
        result = np.dot(input_data, w.astype(dtype.as_numpy_dtype))

      self.assertAllClose(
          result,
          output_data,
          atol=tolerance_map[dtype],
          rtol=tolerance_map[dtype])

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testSharing(self, use_bias):

    np.random.seed(self.seed)
    inp_1 = tf.placeholder(tf.float32, shape=[self.batch_size, self.in_size])
    inp_2 = tf.placeholder(tf.float32, shape=[self.batch_size, self.in_size])

    if use_bias:
      initializers = {"w": _test_initializer(), "b": _test_initializer()}
    else:
      initializers = {"w": _test_initializer()}

    lin = snt.Linear(output_size=self.out_size,
                     use_bias=use_bias,
                     initializers=initializers)
    out_1 = lin(inp_1)
    out_2 = lin(inp_2)
    with self.test_session() as sess:
      # Put the same data into each input, outputs should be identical.
      input_data = np.random.randn(self.batch_size, self.in_size)
      sess.run(tf.global_variables_initializer())
      out_data_1, out_data_2 = sess.run([out_1, out_2],
                                        {inp_1: input_data, inp_2: input_data})
    self.assertAllEqual(out_data_1, out_data_2)

  def testUniquifying(self):
    # Create three modules in same scope with same name - make_template will
    # uniquify them.
    inp = tf.placeholder(tf.float32, shape=[self.batch_size, self.in_size])
    mod_name = "another_linear_module"
    lin1 = snt.Linear(name=mod_name, output_size=self.out_size)
    lin2 = snt.Linear(name=mod_name, output_size=self.out_size)
    lin3 = snt.Linear(name=mod_name, output_size=self.out_size)

    # Connect all the modules to instantiate the variables.
    lin1(inp)
    lin2(inp)
    lin3(inp)

    # Ensure the module name property has been uniquified and is accessible.
    self.assertEqual(lin1.scope_name, mod_name)
    self.assertEqual(lin2.scope_name, mod_name + "_1")
    self.assertEqual(lin3.scope_name, mod_name + "_2")

    self.assertEqual(lin1.module_name, mod_name)
    self.assertEqual(lin2.module_name, mod_name + "_1")
    self.assertEqual(lin3.module_name, mod_name + "_2")

    vars1 = lin1.get_variables()
    vars2 = lin2.get_variables()
    vars3 = lin3.get_variables()

    # Ensure variable names have been made unique.
    for v in vars1:
      self.assertRegexpMatches(v.name, r"{}/[wb]:0".format(lin1.scope_name))
    for v in vars2:
      self.assertRegexpMatches(v.name, r"{}/[wb]:0".format(lin2.scope_name))
    for v in vars3:
      self.assertRegexpMatches(v.name, r"{}/[wb]:0".format(lin3.scope_name))

  def testIsConnected(self):
    bad_inputs = tf.placeholder(tf.float32, shape=[self.batch_size,
                                                   self.in_size,
                                                   self.in_size])
    lin = snt.Linear(output_size=self.out_size)

    self.assertFalse(lin.is_connected)
    # This will raise a snt.IncompatibleShapeError because bad_inputs has
    # too many dimensions.
    try:
      lin(bad_inputs)
    except snt.IncompatibleShapeError:
      pass
    self.assertFalse(lin.is_connected)

  def testUnknownInputSize(self):
    bad_inputs = tf.placeholder(tf.float32, shape=[self.batch_size, None])
    lin = snt.Linear(output_size=self.out_size)

    self.assertFalse(lin.is_connected)

    err = "Input size must be specified at module build time"
    with self.assertRaisesRegexp(snt.IncompatibleShapeError, err):
      lin(bad_inputs)

    self.assertFalse(lin.is_connected)

  def testInvalidInitializationParameters(self):
    with self.assertRaisesRegexp(KeyError, "Invalid initializer keys.*"):
      snt.Linear(
          output_size=self.out_size,
          initializers={"not_w": tf.truncated_normal_initializer(stddev=1.0)})

    err = "Initializer for 'w' is not a callable function"
    with self.assertRaisesRegexp(TypeError, err):
      snt.Linear(output_size=self.out_size,
                 initializers={"w": tf.zeros([1, 2, 3])})

  def testInvalidPartitionerParameters(self):
    with self.assertRaisesRegexp(KeyError, "Invalid partitioner keys.*"):
      snt.Linear(
          output_size=self.out_size,
          partitioners={"not_w": tf.fixed_size_partitioner(num_shards=2)})

    err = "Partitioner for 'w' is not a callable function"
    with self.assertRaisesRegexp(TypeError, err):
      snt.Linear(output_size=self.out_size,
                 partitioners={"w": tf.zeros([1, 2, 3])})

  def testInvalidRegularizationParameters(self):
    with self.assertRaisesRegexp(KeyError, "Invalid regularizer keys.*"):
      snt.Linear(
          output_size=self.out_size,
          regularizers={"not_w": tf.contrib.layers.l1_regularizer(scale=0.5)})

    err = "Regularizer for 'w' is not a callable function"
    with self.assertRaisesRegexp(TypeError, err):
      snt.Linear(output_size=self.out_size,
                 regularizers={"w": tf.zeros([1, 2, 3])})

  def testRegularizersInRegularizationLosses(self):
    inputs = tf.zeros([1, 100])
    w_regularizer = tf.contrib.layers.l1_regularizer(scale=0.5)
    b_regularizer = tf.contrib.layers.l2_regularizer(scale=0.5)
    lin = snt.Linear(output_size=100,
                     regularizers={"w": w_regularizer, "b": b_regularizer})
    lin(inputs)

    regularizers = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    self.assertRegexpMatches(regularizers[0].name, ".*l1_regularizer.*")
    self.assertRegexpMatches(regularizers[1].name, ".*l2_regularizer.*")

  def testClone(self):
    inputs = tf.zeros([1, 100])
    linear = snt.Linear(output_size=self.out_size)
    clone1 = linear.clone()
    clone2 = linear.clone(name="clone2")

    linear(inputs)
    clone1(inputs)
    clone2(inputs)

    all_vars = tf.trainable_variables()
    linear_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES,
        scope=linear.variable_scope.name + "/")
    clone1_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES,
        scope=clone1.variable_scope.name + "/")
    clone2_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES,
        scope=clone2.variable_scope.name + "/")

    self.assertEqual(linear.output_size, clone1.output_size)
    self.assertEqual(linear.module_name + "_clone", clone1.module_name)
    self.assertEqual("clone2", clone2.module_name)
    self.assertEqual(len(all_vars), 3*len(linear_vars))
    self.assertEqual(len(linear_vars), len(clone1_vars))
    self.assertEqual(len(linear_vars), len(clone2_vars))

  @parameterized.NamedParameters(
      ("WithBias", True),
      ("WithoutBias", False))
  def testTranspose(self, use_bias):
    with tf.variable_scope("scope1"):
      linear1 = snt.Linear(output_size=self.out_size,
                           use_bias=use_bias,
                           name="linear")
      linear2 = snt.Linear(output_size=self.out_size,
                           use_bias=use_bias,
                           name="linear")
    with tf.variable_scope("scope2"):
      linear_transpose1 = linear1.transpose()
      linear_transpose2 = linear1.transpose()
      linear_transpose3 = linear2.transpose()

    self.assertEqual("scope1/linear", linear1.scope_name)
    self.assertEqual("linear", linear1.module_name)
    self.assertEqual("scope1/linear_1", linear2.scope_name)
    self.assertEqual("linear_1", linear2.module_name)
    self.assertEqual("scope2/linear_transpose", linear_transpose1.scope_name)
    self.assertEqual("linear_transpose", linear_transpose1.module_name)
    self.assertEqual("scope2/linear_transpose_1", linear_transpose2.scope_name)
    self.assertEqual("linear_transpose_1", linear_transpose2.module_name)
    self.assertEqual("scope2/linear_1_transpose", linear_transpose3.scope_name)
    self.assertEqual("linear_1_transpose", linear_transpose3.module_name)

    input_to_linear = tf.placeholder(tf.float32, shape=[self.batch_size,
                                                        self.in_size])

    err = ("Variables in {} not instantiated yet, __call__ the "
           "module first.".format(linear1.scope_name))
    with self.assertRaisesRegexp(snt.NotConnectedError, err):
      linear_transpose1(input_to_linear)

    linear_transpose1 = linear1.transpose()
    self.assertEqual(linear1.has_bias, linear_transpose1.has_bias)

    linear_out = linear1(input_to_linear)
    linear_transposed_output = linear_transpose1(linear_out)
    self.assertEqual(linear_transposed_output.get_shape(),
                     input_to_linear.get_shape())

  def testGradientColocation(self):
    """Tests a particular device (e.g. gpu, cpu) placement.

    This test ensures that the following device placement is possible:

    * The Linear module is on the gpu,
    * the optimizer is declared to be on the cpu,
    * but when calling minimize on the optimizer, we pass True to
      colocate_gradients_with_ops.

    The test exists because while one may expect tf.matmul(X, w) + b to be
    equivalent to tf.nn.xw_plus_b(X, w, b), with the latter this placement
    results in an InvalidArgumentError.

    Warning: if there is no gpu available to tensorflow this test will be
    skipped with just a warning! This is because the test requires that
    tensorflow has access to a gpu, but often this is not the case.
    """
    if not any(x.device_type == "GPU" for x in device_lib.list_local_devices()):
      tf.logging.warn("Skipping the gradient colocation test as there is no "
                      "gpu available to tensorflow.")
      return
    n_outputs = 5
    n_inputs = 3
    batch_size = 7
    linear = snt.Linear(n_outputs)
    with tf.device("/cpu:*"):
      # Set up data.
      inputs = tf.placeholder(tf.float32, [batch_size, n_inputs])
      labels = tf.to_int64(np.ones((batch_size)))
      # Predictions.
      with tf.device("/gpu:*"):
        outputs = linear(inputs)
      # Calculate the loss.
      cross_entropy = tf.contrib.nn.deprecated_flipped_sparse_softmax_cross_entropy_with_logits(  # pylint: disable=line-too-long
          outputs, labels, name="xentropy")
      loss = tf.reduce_mean(cross_entropy, name="xentropy_mean")
      # Optimizer.
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
      optimizer.minimize(loss, colocate_gradients_with_ops=True)
    init = tf.global_variables_initializer()
    try:
      with self.test_session(force_gpu=True) as sess:
        sess.run(init)
    except tf.errors.InvalidArgumentError as e:
      self.fail("Cannot start the session. Details:\n" + e.message)

  def testPartitioners(self):
    inputs = tf.zeros([1, 100])
    partitioners = {
        "w": tf.variable_axis_size_partitioner(10000),
        "b": tf.variable_axis_size_partitioner(100),
    }
    linear = snt.Linear(100, partitioners=partitioners)
    linear(inputs)

    self.assertEqual(type(linear.w), variables.PartitionedVariable)
    self.assertEqual(type(linear.b), variables.PartitionedVariable)

  @parameterized.NamedParameters(
      ("float16", tf.float16),
      ("float32", tf.float32),
      ("float64", tf.float64))
  def testFloatDataTypeConsistent(self, dtype):
    inputs = tf.placeholder(dtype, [3, 7])
    linear = snt.Linear(11)
    outputs = linear(inputs)
    self.assertEqual(linear.w.dtype.base_dtype, dtype)
    self.assertEqual(linear.b.dtype.base_dtype, dtype)
    self.assertEqual(outputs.dtype.base_dtype, dtype)

  def testIntegerDataTypeFailsWithDefaultInitializers(self):
    dtype = tf.int32
    inputs = tf.placeholder(dtype, [3, 7])
    linear = snt.Linear(11)
    with self.assertRaisesRegexp(ValueError, "Expected floating point type"):
      unused_outputs = linear(inputs)

  def testIntegerDataTypeConsistentWithCustomWeightInitializer(self):
    dtype = tf.int32
    inputs = tf.placeholder(dtype, [3, 7])
    linear = snt.Linear(
        11, initializers={"w": tf.zeros_initializer(dtype=dtype)})
    outputs = linear(inputs)
    self.assertEqual(linear.w.dtype.base_dtype, dtype)
    self.assertEqual(linear.b.dtype.base_dtype, dtype)
    self.assertEqual(outputs.dtype.base_dtype, dtype)


class AddBiasTest(tf.test.TestCase, parameterized.ParameterizedTestCase):

  BATCH_SIZE = 11
  IN_SHAPE = (13, 7, 5)
  OUT_SHAPE = IN_SHAPE

  BIAS_DIMS_PARAMETERS = [
      ("DefaultBiasDims", None, IN_SHAPE),
      ("AllBiasDims", [1, 2, 3], IN_SHAPE),
      ("ScalarBiasDims", [], ()),
      ("LastBiasDims", [-1], (IN_SHAPE[2],)),
      ("ExplicitLastBiasDims", [3], (IN_SHAPE[2],)),
      ("FirstBiasDims", [1], (IN_SHAPE[0], 1, 1)),
      ("MiddleBiasDims", [2], (IN_SHAPE[1], 1)),
  ]

  def setUp(self):
    super(AddBiasTest, self).setUp()
    self.mb_in_shape = (self.BATCH_SIZE,) + self.IN_SHAPE
    self.mb_out_shape = (self.BATCH_SIZE,) + self.OUT_SHAPE
    self.seed = 42

  @parameterized.NamedParameters(*BIAS_DIMS_PARAMETERS)
  def testShape(self, bias_dims, unused_bias_shape):
    inputs = tf.placeholder(tf.float32, shape=self.mb_in_shape)
    add = snt.AddBias(bias_dims=bias_dims)
    output = add(inputs)
    self.assertTrue(
        output.get_shape().is_compatible_with(self.mb_out_shape))

  @parameterized.NamedParameters(*BIAS_DIMS_PARAMETERS)
  def testName(self, bias_dims, unused_bias_shape):
    mod_name = "unique_name"
    with tf.variable_scope("scope"):
      add = snt.AddBias(name=mod_name, bias_dims=bias_dims)
    self.assertEqual(add.scope_name, "scope/" + mod_name)
    self.assertEqual(add.module_name, mod_name)

  @parameterized.NamedParameters(*BIAS_DIMS_PARAMETERS)
  def testVariables(self, bias_dims, bias_shape):
    inputs = tf.placeholder(tf.float32, shape=self.mb_in_shape)
    add = snt.AddBias(bias_dims=bias_dims)

    err = ("Variables in {} not instantiated yet, __call__ "
           "the module first.".format(add.scope_name))
    with self.assertRaisesRegexp(snt.NotConnectedError, err):
      add.get_variables()

    err = ("Variables in {} not instantiated yet, __call__ "
           "the module first.".format(add.scope_name))
    with self.assertRaisesRegexp(snt.NotConnectedError, err):
      _ = add.b

    add(inputs)  # Connect the module, but ignore the return value.

    variables_ = add.get_variables()
    self.assertEqual(len(variables_), 1, "Add should have 1 variable.")

    for v in variables_:
      self.assertRegexpMatches(v.name, r"{}/[b]:0".format(add.scope_name))
      shape = np.ndarray(bias_shape)
      self.assertShapeEqual(shape, v.initial_value)

  @parameterized.NamedParameters(*BIAS_DIMS_PARAMETERS)
  def testComputation(self, bias_dims, bias_shape):
    np.random.seed(self.seed)
    types = (tf.float16, tf.float32, tf.float64)
    tol = (1e-2, 1e-6, 1e-9)
    tolerance_map = dict(zip(types, tol))
    b_regularizer = tf.contrib.layers.l2_regularizer(scale=0.5)
    for dtype in types:
      inputs = tf.placeholder(dtype, shape=self.mb_in_shape)
      add = snt.AddBias(bias_dims=bias_dims,
                        initializers={"b": _test_initializer()},
                        regularizers={"b": b_regularizer})
      output = add(inputs)
      output_subtract = add(inputs, multiplier=-1)
      with self.test_session() as sess:
        # With random data, check the TF calculation matches the Numpy version.
        input_data = np.random.randn(*self.mb_in_shape).astype(
            dtype.as_numpy_dtype)
        sess.run(tf.global_variables_initializer())
        output_data, output_subtract_data, b = sess.run(
            [output, output_subtract, add.b], {inputs: input_data})
        regularizers = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.assertRegexpMatches(regularizers[0].name, ".*l2_regularizer.*")
      if not bias_shape:  # Scalar bias.
        b_array = np.array([b]).astype(dtype.as_numpy_dtype(b))
      else:
        b_array = b.astype(dtype.as_numpy_dtype)
      result = input_data + b_array
      result_subtract = input_data - b_array
      self.assertAllClose(
          result,
          output_data,
          atol=tolerance_map[dtype],
          rtol=tolerance_map[dtype])
      self.assertAllClose(
          result_subtract,
          output_subtract_data,
          atol=tolerance_map[dtype],
          rtol=tolerance_map[dtype])

  @parameterized.NamedParameters(*BIAS_DIMS_PARAMETERS)
  def testSharing(self, bias_dims, unused_bias_shape):

    np.random.seed(self.seed)
    inp_1 = tf.placeholder(tf.float32, shape=self.mb_in_shape)
    inp_2 = tf.placeholder(tf.float32, shape=self.mb_in_shape)
    add = snt.AddBias(bias_dims=bias_dims,
                      initializers={"b": _test_initializer()})
    out_1 = add(inp_1)
    out_2 = add(inp_2)
    with self.test_session() as sess:
      # Put the same data into each input, outputs should be identical.
      input_data = np.random.randn(*self.mb_in_shape)
      sess.run(tf.global_variables_initializer())
      out_data_1, out_data_2 = sess.run([out_1, out_2],
                                        {inp_1: input_data, inp_2: input_data})
    self.assertAllEqual(out_data_1, out_data_2)

  @parameterized.NamedParameters(*BIAS_DIMS_PARAMETERS)
  def testUniquifying(self, bias_dims, unused_bias_shape):
    # Create three modules in same scope with same name - make_template will
    # uniquify them.
    inp = tf.placeholder(tf.float32, shape=self.mb_in_shape)
    mod_name = "another_linear_module"
    add1 = snt.AddBias(bias_dims=bias_dims, name=mod_name)
    add2 = snt.AddBias(bias_dims=bias_dims, name=mod_name)
    add3 = snt.AddBias(bias_dims=bias_dims, name=mod_name)

    # Connect all the modules to instantiate the variables.
    add1(inp)
    add2(inp)
    add3(inp)

    # Ensure the module name property has been uniquified and is accessible.
    self.assertEqual(add1.module_name, mod_name)
    self.assertEqual(add2.module_name, mod_name + "_1")
    self.assertEqual(add3.module_name, mod_name + "_2")

    vars1 = add1.get_variables()
    vars2 = add2.get_variables()
    vars3 = add3.get_variables()

    # Ensure variable names have been made unique.
    for v in vars1:
      self.assertRegexpMatches(v.name, r"{}/[b]:0".format(add1.scope_name))
    for v in vars2:
      self.assertRegexpMatches(v.name, r"{}/[b]:0".format(add2.scope_name))
    for v in vars3:
      self.assertRegexpMatches(v.name, r"{}/[b]:0".format(add3.scope_name))

  @parameterized.NamedParameters(*BIAS_DIMS_PARAMETERS)
  def testInvalidInitializationParameters(self, bias_dims, unused_bias_shape):
    err = "Invalid initializer keys.*"
    with self.assertRaisesRegexp(KeyError, err):
      snt.AddBias(
          bias_dims=bias_dims,
          initializers={"not_b": tf.truncated_normal_initializer(stddev=1.0)})

    err = "Initializer for 'b' is not a callable function"
    with self.assertRaisesRegexp(TypeError, err):
      snt.AddBias(
          bias_dims=bias_dims,
          initializers={"b": tf.zeros([1, 2, 3])})

  @parameterized.NamedParameters(*BIAS_DIMS_PARAMETERS)
  def testInvalidPartitionerParameters(self, bias_dims, unused_bias_shape):
    with self.assertRaisesRegexp(KeyError, "Invalid partitioner keys.*"):
      snt.AddBias(
          bias_dims=bias_dims,
          partitioners={"not_b": tf.fixed_size_partitioner(num_shards=2)})

    err = "Partitioner for 'b' is not a callable function"
    with self.assertRaisesRegexp(TypeError, err):
      snt.AddBias(
          bias_dims=bias_dims,
          partitioners={"b": tf.zeros([1, 2, 3])})

  @parameterized.NamedParameters(*BIAS_DIMS_PARAMETERS)
  def testInvalidRegularizationParameters(self, bias_dims, unused_bias_shape):
    with self.assertRaisesRegexp(KeyError, "Invalid regularizer keys.*"):
      snt.AddBias(
          bias_dims=bias_dims,
          regularizers={"not_b": tf.contrib.layers.l1_regularizer(scale=0.5)})

    err = "Regularizer for 'b' is not a callable function"
    with self.assertRaisesRegexp(TypeError, err):
      snt.AddBias(bias_dims=bias_dims,
                  regularizers={"b": tf.zeros([1, 2, 3])})

  @parameterized.NamedParameters(*BIAS_DIMS_PARAMETERS)
  def testTranspose(self, bias_dims, unused_bias_shape):
    add = snt.AddBias(bias_dims=bias_dims)
    input_to_add = tf.placeholder(tf.float32, shape=self.mb_in_shape)

    # Check error occurs when we build the transposed module before the
    # original.
    add_transpose = add.transpose()
    err = "Build the original untransposed module before building this one."
    with self.assertRaisesRegexp(snt.ParentNotBuiltError, err):
      add_transpose(input_to_add)

    # Check that building the original before the transposed works as intended.
    add_transpose = add.transpose()
    add_out = add(input_to_add)
    add_transpose_out = add_transpose(add_out)
    self.assertEqual(add_transpose_out.get_shape(),
                     input_to_add.get_shape())
    self.assertEqual(add_transpose.b.get_shape(),
                     add.b.get_shape())

  def testPartitioners(self):
    inputs = tf.zeros([1, 100])
    partitioners = {
        "b": tf.variable_axis_size_partitioner(10000),
    }
    bias = snt.AddBias(partitioners=partitioners)
    bias(inputs)

    self.assertEqual(type(bias.b), variables.PartitionedVariable)


class TrainableVariableTest(tf.test.TestCase):

  def testName(self):
    mod_name = "unique_name"
    with tf.variable_scope("scope"):
      mod = snt.TrainableVariable(name=mod_name, shape=[1])
    self.assertEqual(mod.scope_name, "scope/" + mod_name)
    self.assertEqual(mod.module_name, mod_name)

  def testInitialization(self):
    # Checks that the module initialization correctly sets the shape of the
    # internal variable w.
    shape = [1, 2, 3]
    var = snt.TrainableVariable(
        shape=shape,
        dtype=tf.float32,
        initializers={"w": tf.zeros_initializer()})
    # We need to connect the module to the graph in order to inspect its
    # variables
    var()
    self.assertEqual(var.w.get_shape(), shape)

  def testVariableInitialization(self):
    # Check that a simple operation involving the TrainableVariable
    # matches the result of the corresponding operation in numpy
    np.random.seed(100)
    types = (tf.float16, tf.float32, tf.float64)
    tol = (1e-2, 1e-6, 1e-9)
    tolerance_map = dict(zip(types, tol))
    lhs_shape = [3, 4]
    rhs_shape = [4, 6]
    for dtype in types:
      x = tf.placeholder(dtype, shape=lhs_shape)
      var = snt.TrainableVariable(shape=rhs_shape,
                                  dtype=dtype,
                                  initializers={"w": _test_initializer()})
      y = tf.matmul(x, var())
      with self.test_session() as sess:
        lhs_matrix = np.random.randn(*lhs_shape)
        sess.run(tf.global_variables_initializer())
        product, w = sess.run([y, var.w], {x: lhs_matrix})
      self.assertAllClose(product,
                          np.dot(
                              lhs_matrix.astype(dtype.as_numpy_dtype),
                              w.astype(dtype.as_numpy_dtype)),
                          atol=tolerance_map[dtype],
                          rtol=tolerance_map[dtype])

  def testInvalidInitializationParameters(self):
    variable_name = "trainable_variable"
    with self.assertRaisesRegexp(KeyError, "Invalid initializer keys.*"):
      snt.TrainableVariable(
          name=variable_name,
          shape=[1],
          initializers={"w": tf.truncated_normal_initializer(stddev=1.0),
                        "extra": tf.truncated_normal_initializer(stddev=1.0)})

    with self.assertRaisesRegexp(KeyError, "Invalid initializer keys.*"):
      snt.TrainableVariable(
          name=variable_name,
          shape=[1],
          initializers={"not_w": tf.truncated_normal_initializer(stddev=1.0)})

    err = "Initializer for 'w' is not a callable function"
    with self.assertRaisesRegexp(TypeError, err):
      snt.TrainableVariable(name=variable_name,
                            shape=[1],
                            initializers={"w": tf.zeros([1, 2, 3])})

  def testCallBeforeInstantiation(self):
    variable_name = "trainable_variable"
    var = snt.TrainableVariable(name=variable_name, shape=[1])

    err = r"Variables in {} not instantiated yet.*".format(variable_name)
    with self.assertRaisesRegexp(snt.NotConnectedError, err):
      var.get_variables()

    err = r"Variables in {} not instantiated yet.*".format(variable_name)
    with self.assertRaisesRegexp(snt.NotConnectedError, err):
      _ = var.w

  def testInvalidPartitionerParameters(self):
    with self.assertRaisesRegexp(KeyError, "Invalid partitioner keys.*"):
      snt.TrainableVariable(
          shape=[1],
          partitioners={"not_w": tf.fixed_size_partitioner(num_shards=2)})

    err = "Partitioner for 'w' is not a callable function"
    with self.assertRaisesRegexp(TypeError, err):
      snt.TrainableVariable(
          shape=[1],
          partitioners={"w": tf.zeros([1, 2, 3])})

  def testInvalidRegularizationParameters(self):
    variable_name = "trainable_variable"
    with self.assertRaisesRegexp(KeyError, "Invalid regularizer keys.*"):
      snt.TrainableVariable(
          name=variable_name,
          shape=[1],
          regularizers={"not_w": tf.contrib.layers.l1_regularizer(scale=0.5)})

    err = "Regularizer for 'w' is not a callable function"
    with self.assertRaisesRegexp(TypeError, err):
      snt.TrainableVariable(name=variable_name, shape=[1],
                            regularizers={"w": tf.zeros([1, 2, 3])})

  def testRegularizersInRegularizationLosses(self):
    variable_name = "trainable_variable"
    w_regularizer = tf.contrib.layers.l1_regularizer(scale=0.5)
    var = snt.TrainableVariable(
        name=variable_name, shape=[1], regularizers={"w": w_regularizer})
    var()

    regularizers = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    self.assertRegexpMatches(regularizers[0].name, ".*l1_regularizer.*")

  def testPartitioners(self):
    partitioners = {
        "w": tf.variable_axis_size_partitioner(10000),
    }
    var = snt.TrainableVariable(
        shape=[10, 13],
        partitioners=partitioners)
    var()

    self.assertEqual(type(var.w), variables.PartitionedVariable)


class BatchReshapeTest(tf.test.TestCase,
                       parameterized.ParameterizedTestCase):

  def testName(self):
    mod_name = "unique_name"
    with tf.variable_scope("scope"):
      mod = snt.BatchReshape(name=mod_name, shape=[-1])
    self.assertEqual(mod.scope_name, "scope/" + mod_name)
    self.assertEqual(mod.module_name, mod_name)

  def testReshape(self):
    batch_size = 10
    in_shape = [2, 3, 4, 5]
    out_shape = [2 * 3, 5, 4]
    assert np.prod(in_shape) == np.prod(out_shape)
    inputs = tf.placeholder(tf.float32, shape=[batch_size] + in_shape)
    mod = snt.BatchReshape(shape=out_shape)
    output = mod(inputs)
    self.assertEqual(output.get_shape(), [batch_size] + out_shape)

  def testInvalidReshapeParameters(self):
    batch_size = 10
    in_shape = [2, 3, 4, 5]
    inputs = tf.placeholder(tf.float32, shape=[batch_size] + in_shape)
    # Shape array has invalid format
    err = "Wildcard -1 can appear only once in desired output shape. "
    with self.assertRaisesRegexp(ValueError, err):
      output_invalid_shape_format = [-1, -1]
      snt.BatchReshape(shape=output_invalid_shape_format)(inputs)

    err = ("Desired shape can only contain positive integral numbers "
           "and the wildcard -1. ")
    with self.assertRaisesRegexp(ValueError, err):
      output_invalid_shape_format = [2, 3, -2]
      snt.BatchReshape(shape=output_invalid_shape_format)(inputs)

    # Shape array contains invalid entries
    err = ("Desired shape can only contain positive integral numbers "
           "and the wildcard -1. ")
    with self.assertRaisesRegexp(ValueError, err):
      invalid_shape_type = [7, "string"]
      snt.BatchReshape(shape=invalid_shape_type)(inputs)

    # Incompatible input and output shapes
    err = "Output shape is incompatible with input shape"
    with self.assertRaisesRegexp(ValueError, err):
      out_shape = [2 * 2, 5, 4]
      snt.BatchReshape(shape=out_shape)(inputs)

    # Checks the 2D case.
    with self.assertRaisesRegexp(ValueError, err):
      snt.BatchReshape(shape=[batch_size, 1])(tf.zeros([batch_size, 2]))

  def testCallable(self):
    inputs = tf.placeholder(tf.float32, shape=[2, 3])
    out_shape_lambda = lambda: [3]
    mod = snt.BatchReshape(shape=out_shape_lambda)
    output = mod(inputs)
    self.assertEqual(output.get_shape(), [2, 3])

  def testInferShape(self):
    batch_size = 10
    in_shape = [2, 3, 4, 5]
    out_size = [2, -1, 5]
    correct_out_size = [2, 3 * 4, 5]
    inputs = tf.placeholder(tf.float32, shape=[batch_size] + in_shape)
    mod = snt.BatchReshape(shape=out_size)
    output = mod(inputs)
    self.assertEqual(output.get_shape(), [batch_size] + correct_out_size)

  def testAddDimensions(self):
    batch_size = 10
    in_shape = []
    out_size = [1, 1]
    correct_out_size = [1, 1]
    inputs = tf.placeholder(tf.float32, shape=[batch_size] + in_shape)
    mod = snt.BatchReshape(shape=out_size)
    output = mod(inputs)
    self.assertEqual(output.get_shape(), [batch_size] + correct_out_size)
    # Transposition should also work
    mod_t = mod.transpose()
    t_output = mod_t(output)
    self.assertEqual(t_output.get_shape(), [batch_size] + in_shape)

  def testNoReshapeNeeded(self):
    batch_size = 10
    in_shape = [None]
    out_size = [-1]
    inputs = tf.placeholder(tf.float32, shape=[batch_size] + in_shape)
    mod = snt.BatchReshape(shape=out_size)
    output = mod(inputs)
    self.assertIs(output, inputs)

    in_shape = [10]
    out_size = [10]
    inputs = tf.placeholder(tf.float32, shape=[batch_size] + in_shape)
    mod = snt.BatchReshape(shape=out_size)
    output = mod(inputs)
    self.assertIs(output, inputs)

  @parameterized.NamedParameters(
      ("BadUnknown1", (None,), (5,)),
      ("BadUnknown2", (None, None), (5,)),
      ("BadUnknown3", (None, None), (5, 5)),
      ("BadUnknown4", (5, None), (5, 5)),
      ("BadUnknown5", (None, 5), (5, 5)),
  )
  def testBadUnknownNonPreservedDimensions(self, input_shape, output_shape):
    preserved_shape = (10,)
    shape = preserved_shape + input_shape
    preserve_dims = len(preserved_shape)
    inputs = tf.placeholder(tf.float32, shape)
    mod = snt.BatchReshape(shape=output_shape,
                           preserve_dims=preserve_dims)
    err = "Unknown non-preserved dimensions are not allowed"
    with self.assertRaisesRegexp(ValueError, err):
      _ = mod(inputs)

  def testFlatten(self):
    batch_size = 10
    in_shape = [2, 3, 4, 5]
    out_size = [-1]
    inputs = tf.placeholder(tf.float32, shape=[batch_size] + in_shape)
    mod = snt.BatchReshape(shape=out_size)
    output = mod(inputs)
    flattened_shape = np.prod(in_shape)
    self.assertEqual(output.get_shape(), [batch_size, flattened_shape])

  def testUnknown(self):
    batch_size = None
    in_shape = [2, 3, 4, 5]
    out_size = [-1]
    inputs = tf.placeholder(tf.float32, shape=[batch_size] + in_shape)
    mod = snt.BatchReshape(shape=out_size)
    output = mod(inputs)
    flattened_shape = np.prod(in_shape)
    self.assertEqual(output.get_shape().as_list(),
                     [batch_size, flattened_shape])

  def testTranspose(self):
    batch_size = 10
    in_shape = [2, 3, 4, 5]
    out_size = [2, -1, 5]
    correct_out_size = [2, 3 * 4, 5]
    inputs = tf.random_uniform(shape=[batch_size] + in_shape)
    mod = snt.BatchReshape(shape=out_size)
    mod_t = mod.transpose()
    mod_t_t = mod_t.transpose()
    intermediate_output = mod(inputs)
    self.assertEqual(intermediate_output.get_shape(),
                     [batch_size] + correct_out_size)
    output = mod_t(intermediate_output)
    self.assertEqual(output.get_shape(), [batch_size] + in_shape)
    further_output = mod_t_t(output)
    self.assertEqual(further_output.get_shape(),
                     [batch_size] + correct_out_size)
    with self.test_session() as sess:
      input_data, out = sess.run([inputs, output])
      self.assertAllClose(out, input_data)

  def testInvalidPreserveDimsError(self):
    with self.assertRaisesRegexp(ValueError, "preserve_dims"):
      snt.BatchReshape((-1,), preserve_dims=0)

  def testBuildDimError(self):
    mod = snt.BatchReshape((-1,), preserve_dims=2)
    input_tensor = tf.placeholder(tf.float32, (50,))
    with self.assertRaisesRegexp(ValueError, "preserve_dims"):
      mod(input_tensor)

  def testBuildUnknown(self):
    mod = snt.BatchReshape(shape=(2, 9), preserve_dims=2)
    shape = [50, None, 6, 3]
    inputs = tf.placeholder(tf.float32, shape)
    output = mod(inputs)
    self.assertEqual(output.get_shape().as_list(), [50, None, 2, 9])

  @parameterized.NamedParameters(
      ("Preserve1", (1,)),
      ("Preserve24", (2, 4)),
      ("Preserve?", (None,)),
      ("Preserve?5", (None, 5)),
      ("Preserve5?", (5, None)),
      ("Preserve??", (None, None)))
  def testPreserve(self, preserve):
    shape = list(preserve) + [13, 84, 3, 2]
    output_shape = [13, 21, 3, 8]
    preserve_dims = len(preserve)
    inputs = tf.placeholder(tf.float32, shape)
    mod = snt.BatchReshape(shape=output_shape,
                           preserve_dims=preserve_dims)
    output = mod(inputs)
    self.assertEqual(output.get_shape().as_list(),
                     list(preserve) + output_shape)

  @parameterized.NamedParameters(
      ("Session1", (1,), (2, 3), (-1,)),
      ("Session2", (1, 7), (2, 3), (-1,)),
      ("Session3", (None,), (2, 3), (-1,)),
      ("Session4", (None, 5, None), (2, 3, 4), (4, 6)),
      ("Session5", (None, None, None), (2, 3, 4), (-1,)),
      ("Session6", (5, None, None), (1, 3, 1), (-1,)),
      ("Session7", (1,), (4, 3), (2, 2, 1, 3)),
      ("Session8", (None,), (4, 3), (2, 2, 1, 3)),
      ("Session9", (1, None, 5, None), (4, 3), (2, 2, -1, 3)))
  def testRun(self, preserve, trailing_in, trailing_out):
    rng = np.random.RandomState(0)
    input_shape = preserve + trailing_in
    output_shape = preserve + np.zeros(trailing_in).reshape(trailing_out).shape
    inputs = tf.placeholder(tf.float32, input_shape)
    mod = snt.BatchReshape(shape=trailing_out,
                           preserve_dims=len(preserve))
    output = mod(inputs)
    self.assertEqual(output.get_shape().as_list(), list(output_shape))

    actual_input_shape = [13 if i is None else i for i in input_shape]
    expected_output_shape = [13 if i is None else i for i in output_shape]
    actual_input = rng.rand(*actual_input_shape).astype(np.float32)
    expected_output = actual_input.reshape(expected_output_shape)
    with self.test_session() as sess:
      actual_output = sess.run(output, feed_dict={inputs: actual_input})
    self.assertAllEqual(actual_output, expected_output)


class BatchFlattenTest(tf.test.TestCase,
                       parameterized.ParameterizedTestCase):

  def testName(self):
    mod_name = "unique_name"
    with tf.variable_scope("scope"):
      mod = snt.BatchFlatten(name=mod_name)
    self.assertEqual(mod.scope_name, "scope/" + mod_name)
    self.assertEqual(mod.module_name, mod_name)

  def testFlatten(self):
    batch_size = 10
    in_shape = [2, 3, 4, 5]
    inputs = tf.placeholder(tf.float32, shape=[batch_size] + in_shape)
    mod = snt.BatchFlatten()
    output = mod(inputs)
    flattened_size = np.prod(in_shape)
    self.assertEqual(output.get_shape(), [batch_size, flattened_size])

  @parameterized.Parameters(1, 2, 3, 4)
  def testPreserveDimsOk(self, preserve_dims):
    in_shape = [10, 2, 3, 4]
    inputs = tf.placeholder(tf.float32, shape=in_shape)
    mod = snt.BatchFlatten(preserve_dims=preserve_dims)
    output = mod(inputs)
    flattened_shape = (in_shape[:preserve_dims] +
                       [np.prod(in_shape[preserve_dims:])])
    self.assertEqual(output.get_shape(), flattened_shape)

  @parameterized.Parameters(5, 6, 7, 10)
  def testPreserveDimsError(self, preserve_dims):
    in_shape = [10, 2, 3, 4]
    inputs = tf.placeholder(tf.float32, shape=in_shape)
    err = "Input tensor has 4 dimensions"
    mod = snt.BatchFlatten(preserve_dims=preserve_dims)
    with self.assertRaisesRegexp(ValueError, err):
      _ = mod(inputs)

  def testFlattenWithZeroDim(self):
    inputs = tf.placeholder(tf.float32, shape=[1, 0])
    output = snt.BatchFlatten()(inputs)
    self.assertEqual(output.get_shape(), [1, 0])


class FlattenTrailingDimensionsTest(tf.test.TestCase,
                                    parameterized.ParameterizedTestCase):

  def testName(self):
    mod_name = "unique_name"
    with tf.variable_scope("scope"):
      mod = snt.FlattenTrailingDimensions(dim_from=2, name=mod_name)
    self.assertEqual(mod.scope_name, "scope/" + mod_name)
    self.assertEqual(mod.module_name, mod_name)

  def testInvalidFlattenFromError(self):
    with self.assertRaisesRegexp(ValueError, "dim_from"):
      snt.FlattenTrailingDimensions(dim_from=0)

  def testBuildDimError(self):
    mod = snt.FlattenTrailingDimensions(dim_from=2)
    input_tensor = tf.placeholder(tf.float32, (50,))
    with self.assertRaisesRegexp(ValueError, "dim_from"):
      mod(input_tensor)

  def testBuildUnknown(self):
    mod = snt.FlattenTrailingDimensions(dim_from=2)
    shape = [50, None, 5]
    inputs = tf.placeholder(tf.float32, shape)
    output = mod(inputs)
    self.assertEqual(output.get_shape().as_list(), shape)

  @parameterized.NamedParameters(
      ("BatchSize1", 1),
      ("BatchSize5", 5),
      ("BatchSize?", None))
  def testFlatten(self, batch_size):
    shape = [batch_size, 5, 84, 84, 3, 2]
    inputs = tf.placeholder(tf.float32, shape)
    for dim_from in xrange(1, len(shape)):
      mod = snt.FlattenTrailingDimensions(dim_from)
      output = mod(inputs)
      trailing = np.prod(shape[dim_from:])
      self.assertEqual(output.get_shape().as_list(),
                       shape[:dim_from] + [trailing])

  @parameterized.NamedParameters(
      ("BatchSize1", 1),
      ("BatchSize5", 5),
      ("BatchSize?", None))
  def testTranspose(self, batch_size):
    mod = snt.FlattenTrailingDimensions(dim_from=4)
    mod_trans = mod.transpose()
    initial_shape = [batch_size, 5, 84, 84, 3, 2]
    original = tf.placeholder(tf.float32, initial_shape)
    flat = mod(original)
    self.assertEqual(flat.get_shape().as_list(), initial_shape[:4] + [6])
    final = mod_trans(flat)
    self.assertEqual(final.get_shape().as_list(), initial_shape)


class BatchApplyTest(tf.test.TestCase, parameterized.ParameterizedTestCase):

  def testName(self):
    mod_name = "unique_name"
    with tf.variable_scope("scope"):
      mod = snt.BatchApply(name=mod_name, module_or_op=snt.Linear(2))
    self.assertEqual(mod.scope_name, "scope/" + mod_name)
    self.assertEqual(mod.module_name, mod_name)

  @parameterized.Parameters(False, True)
  def testInferShape(self, test_with_none):
    if test_with_none:
      in_shape = [2, None, 4]
    else:
      in_shape = [2, 3, 4]
    hidden_size = 5
    out_shape1 = in_shape[:2] + [hidden_size]
    out_shape2 = in_shape
    inputs = tf.placeholder(tf.float32, shape=in_shape)
    linear = snt.Linear(hidden_size)
    merge_linear = snt.BatchApply(module_or_op=linear)
    outputs1 = merge_linear(inputs)
    self.assertEqual(outputs1.get_shape().as_list(), out_shape1)
    merge_tanh = snt.BatchApply(module_or_op=tf.tanh)
    outputs2 = merge_tanh(inputs)
    self.assertEqual(outputs2.get_shape().as_list(), out_shape2)

  def testComputation(self):
    np.random.seed(100)
    in_shape = [2, 3, 4]
    in_shape_flat = [6, 4]
    hidden_size = 5
    out_shape1 = in_shape[:2] + [hidden_size]
    out_shape2 = in_shape
    inputs = tf.random_uniform(shape=in_shape)
    inputs_flat = tf.reshape(inputs, shape=in_shape_flat)
    linear = snt.Linear(hidden_size,
                        initializers={"w": _test_initializer(),
                                      "b": _test_initializer()})
    merge_linear = snt.BatchApply(module_or_op=linear)
    outputs1 = merge_linear(inputs)
    outputs1_flat = linear(inputs_flat)
    merge_tanh = snt.BatchApply(module_or_op=tf.tanh)
    outputs2 = merge_tanh(inputs)
    outputs2_flat = merge_tanh(inputs_flat)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      out1, out_flat1 = sess.run([outputs1, outputs1_flat])
      out2, out_flat2 = sess.run([outputs2, outputs2_flat])
      self.assertAllClose(out1, out_flat1.reshape(out_shape1))
      self.assertAllClose(out2, out_flat2.reshape(out_shape2))

  def testVariables(self):
    hidden_size = 5
    in_shape = [2, 3, 4]
    inputs = tf.placeholder(tf.float32, shape=in_shape)
    linear = snt.Linear(hidden_size)
    merge_linear = snt.BatchApply(module_or_op=linear)
    merge_tanh = snt.BatchApply(module_or_op=tf.tanh)
    merge_linear(inputs)
    merge_tanh(inputs)

    # BatchApply doesn't contain any variables inside scope.
    self.assertEqual(merge_linear.get_variables(), ())
    self.assertEqual(merge_tanh.get_variables(), ())

  def testOverTwoDims(self):
    hidden_size = 42
    in_shape = (3, 4, 5, 6)
    expected_out_shape = in_shape[:-1] + (hidden_size,)
    inputs = tf.placeholder(tf.float32, shape=in_shape)
    linear = snt.Linear(output_size=hidden_size)
    merge_linear = snt.BatchApply(module_or_op=linear, n_dims=3)
    output = merge_linear(inputs)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      out_np = sess.run(output, {inputs: np.random.randn(*in_shape)})
      self.assertEqual(expected_out_shape, out_np.shape)

  def testDifferentOutputStructure(self):
    in1 = np.random.randn(3, 5, 7)
    in2 = np.random.randn(3, 5, 11, 8)
    inputs = [tf.constant(in1), tf.constant(in2)]

    def build(inputs):
      a, b = inputs
      a.get_shape().assert_is_compatible_with([3 * 5, 7])
      b.get_shape().assert_is_compatible_with([3 * 5, 11, 8])
      return b
    op = snt.Module(build)
    module = snt.BatchApply(op)
    output = module(inputs)

    with self.test_session() as sess:
      out_np = sess.run(output)
      self.assertAllEqual(in2, out_np)

  def testNested(self):
    # Make a complicated nested input, where we want to flatten the first
    # dimensions of each Tensor before applying
    ab_tuple = collections.namedtuple("ab_tuple", "a, b")
    ab = ab_tuple(a=tf.placeholder(tf.float32, shape=[3, 4, 5]),
                  b=(tf.placeholder(tf.float32, shape=[3, 4, 7]),
                     tf.placeholder(tf.float32, shape=[3, 4, 8])))

    class SizeChecker(snt.AbstractModule):
      """Dummy module checking input is correct structure & size."""

      def __init__(self, tester, name="size_checker"):
        super(SizeChecker, self).__init__(name=name)
        self._tester = tester

      def _build(self, inputs):
        # Structure of the nesting should be the same, even though the Tensors
        # will have been reshaped at this point.
        snt.nest.assert_same_structure(ab, inputs)

        self._tester.assertListEqual(inputs.a.get_shape().as_list(), [12, 5])
        self._tester.assertListEqual(inputs.b[0].get_shape().as_list(), [12, 7])
        self._tester.assertListEqual(inputs.b[1].get_shape().as_list(), [12, 8])

        return inputs  # Return the inputs unmodified

    output = snt.BatchApply(module_or_op=SizeChecker(self), n_dims=2)(ab)

    snt.nest.assert_same_structure(output, ab)
    self.assertShapeEqual(np.zeros((3, 4, 5)), output.a)
    self.assertShapeEqual(np.zeros((3, 4, 7)), output.b[0])
    self.assertShapeEqual(np.zeros((3, 4, 8)), output.b[1])

  def testInputExampleIndex(self):
    in1 = tf.random_normal((3, 5))
    in2 = tf.random_normal((3, 9))

    def build(inputs):
      a, b = inputs
      a.get_shape().assert_is_compatible_with([3 * 5])
      b.get_shape().assert_is_compatible_with([3 * 9])
      return b

    op = snt.Module(build)

    # Checks an error is thrown when the input example contains a different
    # shape for the leading dimensions as the output.
    with self.assertRaises(ValueError):
      snt.BatchApply(op, n_dims=2, input_example_index=0)((in1, in2))

    # Check correct operation when the specified input example contains the same
    # shape for the leading dimensions as the output.
    output = snt.BatchApply(op, n_dims=2, input_example_index=1)((in1, in2))
    with self.test_session() as sess:
      in2_np, out_np = sess.run([in2, output])
      self.assertAllEqual(in2_np, out_np)

  def testMultipleArgs(self):
    in1 = np.random.randn(2, 3, 4, 5)
    in2 = np.random.randn(2, 3, 5, 8)

    module = snt.BatchApply(tf.matmul)
    output = module(in1, in2)
    output.get_shape().assert_is_compatible_with([2, 3, 4, 8])

    expected_output = tf.matmul(in1, in2)
    with self.test_session() as sess:
      out_expected, out_result = sess.run([expected_output, output])
      self.assertAllClose(out_expected, out_result)

  def testKWArgs(self):
    in1 = np.random.randn(2, 3, 4, 5)
    in2 = np.random.randn(2, 3, 5, 8)

    module = snt.BatchApply(tf.matmul)
    output = module(a=in1, b=in2)
    output.get_shape().assert_is_compatible_with([2, 3, 4, 8])

    expected_output = tf.matmul(in1, in2)
    with self.test_session() as sess:
      out_expected, out_result = sess.run([expected_output, output])
      self.assertAllClose(out_expected, out_result)

  def testHandlesReturnedNone(self):
    def fn(input_):
      del input_
      return None
    result = snt.BatchApply(fn)(tf.zeros([1, 1]))
    self.assertEqual(result, None)

  def testSomeInputsAreNone(self):
    in1 = np.random.randn(2, 3, 4, 5)
    in2 = np.random.randn(2, 3, 5, 8)
    in3 = None

    def build(input1, input2, input3):
      output = tf.matmul(input1, input2)
      if input3 is not None:
        output = tf.matmul(input3)
      return output

    module = snt.BatchApply(build)
    output = module(in1, in2, in3)
    output.get_shape().assert_is_compatible_with([2, 3, 4, 8])

    expected_output = tf.matmul(in1, in2)
    with self.test_session() as sess:
      out_expected, out_result = sess.run([expected_output, output])
    self.assertAllClose(out_expected, out_result)


class SliceByDimTest(tf.test.TestCase):

  def testName(self):
    mod_name = "unique_name"
    with tf.variable_scope("scope"):
      mod = snt.SliceByDim(name=mod_name, dims=[0, 2],
                           begin=[0, 0], size=[2, 4])
    self.assertEqual(mod.scope_name, "scope/" + mod_name)
    self.assertEqual(mod.module_name, mod_name)

  def testInferShape(self):
    in_shape = [2, 3, 4, 5, 6]
    dims = [0, 2, 4]
    begin = [0, 1, 2]
    size = [1, 2, 3]
    out_shape = [1, 3, 2, 5, 3]
    inputs = tf.placeholder(tf.float32, shape=in_shape)
    mod = snt.SliceByDim(dims=dims, begin=begin, size=size)
    output = mod(inputs)
    self.assertEqual(output.get_shape(), out_shape)

  def testComparison(self):
    # Here we compare the output with the tf.slice equivalent.
    in_shape = [2, 3, 4]
    inputs = tf.random_uniform(shape=in_shape)

    dims = [0, 2]
    begin = [1, 2]
    size = [1, 2]
    mod = snt.SliceByDim(dims=dims, begin=begin, size=size)
    output = mod(inputs)

    begin_tf = [1, 0, 2]
    size_tf = [1, -1, 2]
    ref_output = tf.slice(inputs, begin=begin_tf, size=size_tf)

    with self.test_session() as sess:
      actual, expected = sess.run([output, ref_output])
      self.assertAllEqual(actual, expected)

  def testComputation(self):
    inputs = tf.constant(dtype=tf.int32, value=[[1, 2, 3], [1, 2, 3]])

    dims = [0, 1]
    begin = [0, 1]
    size = [1, 2]
    mod = snt.SliceByDim(dims=dims, begin=begin, size=size)
    output = mod(inputs)

    with self.test_session() as sess:
      actual = sess.run(output)
      expected = [[2, 3]]
      self.assertAllEqual(actual, expected)

  def testNegativeDim(self):
    inputs = tf.constant(dtype=tf.int32, value=[[1, 2, 3], [4, 5, 6]])

    dims = [0, -1]
    begin = [0, 1]
    size = [-1, 2]
    mod = snt.SliceByDim(dims=dims, begin=begin, size=size)
    output = mod(inputs)

    with self.test_session() as sess:
      actual = sess.run(output)
      expected = [[2, 3], [5, 6]]
      self.assertAllEqual(actual, expected)

  def testInvalidSliceParameters(self):
    dims = [0, 2, 4]
    begin = [0, 0, 0]
    size = [1, 2, 3]

    err = "begin must have the same length as dims: {}.".format(len(dims))
    with self.assertRaisesRegexp(ValueError, err):
      invalid_begin_format = [0, 0]
      _ = snt.SliceByDim(
          dims=dims, begin=invalid_begin_format, size=size)

    err = "size must have the same length as dims: {}.".format(len(dims))
    with self.assertRaisesRegexp(ValueError, err):
      invalid_size_format = [1, 2, 3, 4]
      _ = snt.SliceByDim(
          dims=dims, begin=begin, size=invalid_size_format)

  def testInvalidTensorRank(self):
    dims = [0, 2, 4]
    begin = [0, 0, 0]
    size = [1, 2, 3]
    mod = snt.SliceByDim(dims=dims, begin=begin, size=size)

    in_shape = [2, 3, 4, 5]
    inputs = tf.placeholder(tf.float32, shape=in_shape)

    err = "Rank of inputs must be at least {}.".format(np.max(dims) + 1)
    with self.assertRaisesRegexp(ValueError, err):
      _ = mod(inputs)

  def testUniqueDimensions(self):
    dims = [0, 0, 1]
    begin = [0, 0, 0]
    size = [1, 2, 3]

    err = "dims must not have any repeated integers."
    with self.assertRaisesRegexp(ValueError, err):
      _ = snt.SliceByDim(dims=dims, begin=begin, size=size)


class TileByDimTest(tf.test.TestCase):

  def testName(self):
    mod_name = "unique_name"
    with tf.variable_scope("scope"):
      mod = snt.TileByDim(name=mod_name, dims=[0, 2], multiples=[1, 2])
    self.assertEqual(mod.scope_name, "scope/" + mod_name)
    self.assertEqual(mod.module_name, mod_name)

  def testInferShape(self):
    in_shape = [2, 3, 4, 5, 6]
    dims = [0, 2, 4]
    multiples = [1, 2, 3]
    out_shape = [2, 3, 8, 5, 18]
    inputs = tf.placeholder(tf.float32, shape=in_shape)
    mod = snt.TileByDim(dims=dims, multiples=multiples)
    output = mod(inputs)
    self.assertEqual(output.get_shape(), out_shape)

  def testComparison(self):
    # Here we compare the output with the `tf.tile` equivalent.
    in_shape = [2, 3, 4]
    inputs = tf.random_uniform(shape=in_shape)

    dims = [0, 2]
    multiples = [2, 4]
    mod = snt.TileByDim(dims=dims, multiples=multiples)
    output = mod(inputs)

    multiple_tf = [2, 1, 4]
    ref_output = tf.tile(inputs, multiples=multiple_tf)

    with self.test_session() as sess:
      actual, expected = sess.run([output, ref_output])
      self.assertAllEqual(actual, expected)

  def testComputation(self):
    inputs = tf.constant(dtype=tf.int32, value=[[1, 2, 3], [1, 2, 3]])

    dims = [1]
    multiples = [2]
    mod = snt.TileByDim(dims=dims, multiples=multiples)
    output = mod(inputs)

    with self.test_session() as sess:
      actual = sess.run(output)
      expected = [[1, 2, 3, 1, 2, 3], [1, 2, 3, 1, 2, 3]]
      self.assertAllEqual(actual, expected)

  def testInvalidTileParameters(self):
    dims = [0, 2, 4]
    invalid_multiples_format = [1, 2]

    err = "multiples must have the same length as dims: {}.".format(len(dims))
    with self.assertRaisesRegexp(ValueError, err):
      snt.TileByDim(dims=dims, multiples=invalid_multiples_format)

  def testUniqueDimensions(self):
    dims = [0, 0, 1]
    multiples = [1, 2, 3]

    err = "dims must not have any repeated integers."
    with self.assertRaisesRegexp(ValueError, err):
      snt.TileByDim(dims=dims, multiples=multiples)


class MergeDimsTest(tf.test.TestCase):

  def testName(self):
    mod_name = "unique_name"
    with tf.variable_scope("scope"):
      mod = snt.MergeDims(name=mod_name, start=0, size=2)
    self.assertEqual(mod.scope_name, "scope/" + mod_name)
    self.assertEqual(mod.module_name, mod_name)

  def testInferShape(self):
    in_shape = [2, 3, 4, 5, 6]
    start = 1
    size = 3
    out_shape = [2, 3 * 4 * 5, 6]
    inputs = tf.placeholder(tf.float32, shape=in_shape)
    mod = snt.MergeDims(start=start, size=size)
    output = mod(inputs)
    self.assertEqual(output.get_shape(), out_shape)

  def testComputation(self):
    # Here we compare the output with the tf.reshape equivalent.
    in_shape = [2, 3, 4, 5, 6]
    inputs = tf.random_uniform(shape=in_shape)

    start = 1
    size = 2
    mod = snt.MergeDims(start=start, size=size)
    output = mod(inputs)

    ref_output = tf.reshape(inputs, shape=[2, 3 * 4, 5, 6])

    with self.test_session() as sess:
      out = sess.run([output, ref_output])
      self.assertAllEqual(out[0], out[1])

  def testInvalidDimsParameters(self):
    start = 3
    invalid_size = 1

    err = "`size` should be strictly greater than 1."
    with self.assertRaisesRegexp(ValueError, err):
      snt.MergeDims(start=start, size=invalid_size)

  def testInvalidTensorRank(self):
    start = 0
    size = 4
    mod = snt.MergeDims(start=start, size=size)

    in_shape = [2, 3, 4]
    inputs = tf.placeholder(tf.float32, shape=in_shape)

    err = "Rank of inputs must be at least {}.".format(start + size)
    with self.assertRaisesRegexp(ValueError, err):
      mod(inputs)

  def testNestedInput(self):
    start = 0
    size = 2
    mod = snt.MergeDims(start=start, size=size)

    namedtuple_type = collections.namedtuple("abc", ["a", "b", "c"])
    nested_tensors = [
        tf.random_uniform(shape=[3, 4, 5, 44]),
        [
            tf.random_uniform(shape=[101, 3]),
            tf.random_uniform(shape=[4, 5, 123, 87]),
        ],
        [
            [tf.random_uniform(shape=[1, 2, 3, 4, 5])],
        ],
        namedtuple_type(a=tf.random_uniform(shape=[3, 2, 1]),
                        b=tf.random_uniform(shape=[6, 8, 10, 12]),
                        c=tf.random_uniform(shape=[20, 10]))
    ]

    merged_tensors = mod(nested_tensors)

    nest.assert_same_structure(nested_tensors, merged_tensors)

    for original_tensor, merged_tensor in zip(nest.flatten(nested_tensors),
                                              nest.flatten(merged_tensors)):
      original_shape = original_tensor.get_shape()
      merged_shape = merged_tensor.get_shape()
      self.assertEqual(original_shape.ndims - (size - 1),
                       merged_shape.ndims)
      self.assertEqual(np.prod(original_shape[start:start + size]),
                       merged_shape[start])
      self.assertEqual(original_shape.num_elements(),
                       merged_shape.num_elements())


class SelectInputTest(tf.test.TestCase):

  def testName(self):
    mod_name = "unique_name"
    with tf.variable_scope("scope"):
      mod = snt.SelectInput(name=mod_name, idx=0)
    self.assertEqual(mod.scope_name, "scope/" + mod_name)
    self.assertEqual(mod.module_name, mod_name)

  def testBasicSelect(self):
    """Test where idx is an integer."""
    shape0 = [2, 3]
    shape1 = [2, 3, 4]
    input0 = tf.random_uniform(shape=shape0)
    input1 = tf.random_uniform(shape=shape1)

    mod = snt.SelectInput(idx=0)
    output = mod(input0, input1)
    output0 = tf.identity(input0)

    with self.test_session() as sess:
      out = sess.run([output, output0])
      self.assertAllEqual(out[0], out[1])

  def testTupleSelect(self):
    """Test where idx is a tuple."""
    shape0 = [1, 2]
    shape1 = [1, 2, 3]
    shape2 = [1, 2, 3, 4]
    input0 = tf.random_uniform(shape=shape0)
    input1 = tf.random_uniform(shape=shape1)
    input2 = tf.random_uniform(shape=shape2)

    mod = snt.SelectInput(idx=(0, 2))
    output = mod(input0, input1, input2)
    output0 = tf.identity(input0)
    output2 = tf.identity(input2)

    with self.test_session() as sess:
      out = sess.run([output, [output0, output2]])
      self.assertAllEqual(out[0][0], out[1][0])
      self.assertAllEqual(out[0][1], out[1][1])

  def testNestedListSelect(self):
    """Test where idx is a nested list."""
    shape0 = [1, 2]
    shape1 = [1, 2, 3]
    shape2 = [1, 2, 3, 4]
    input0 = tf.random_uniform(shape=shape0)
    input1 = tf.random_uniform(shape=shape1)
    input2 = tf.random_uniform(shape=shape2)

    mod = snt.SelectInput(idx=[2, [1, 0, 1]])
    output = mod(input0, input1, input2)
    output0 = tf.identity(input0)
    output1 = tf.identity(input1)
    output2 = tf.identity(input2)

    with self.test_session() as sess:
      out = sess.run([output, [output2, [output1, output0, output1]]])
      self.assertAllEqual(out[0][0], out[1][0])
      self.assertAllEqual(out[0][1][0], out[1][1][0])
      self.assertAllEqual(out[0][1][1], out[1][1][1])
      self.assertAllEqual(out[0][1][2], out[1][1][2])

  def testInvalidIdxValue(self):
    """Checks error on invalid idx value."""
    input1 = tf.placeholder(tf.float32, shape=[2, 3, 4, 5, 6])
    input2 = tf.placeholder(tf.float32, shape=[7, 8])

    invalid_idx = 2
    mod = snt.SelectInput(idx=[invalid_idx])

    err = (r"`idx` contains out of bound entries \(they should be in the "
           r"range \[0, 2\)\)")
    with self.assertRaisesRegexp(ValueError, err):
      mod(input1, input2)

  def testInvalidIdxType(self):
    """Checks error on invalid idx type."""
    invalid_idx = 0.5

    err = r"`idx` should be a \(nested\) array/tuple, or an integer."
    with self.assertRaisesRegexp(TypeError, err):
      snt.SelectInput(idx=invalid_idx)


if __name__ == "__main__":
  tf.test.main()
