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

"""Tests sonnet.python.modules.nets.mlp."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import sonnet as snt
from sonnet.testing import parameterized
import tensorflow as tf


class MLPTest(parameterized.ParameterizedTestCase,
              tf.test.TestCase):

  def setUp(self):
    super(MLPTest, self).setUp()

    self.output_sizes = [11, 13, 17]
    self.batch_size = 5
    self.input_size = 7
    self.module_name = "mlp"
    self.initializers = {
        "w": tf.truncated_normal_initializer(stddev=1.0),
    }
    self.regularizers = {
        "w": tf.contrib.layers.l1_regularizer(scale=0.1),
    }
    self.partitioners = {
        "w": tf.fixed_size_partitioner(num_shards=2),
    }

  def testName(self):
    unique_name = "unique_name"
    with tf.variable_scope("scope"):
      mlp = snt.nets.MLP(name=unique_name, output_sizes=self.output_sizes)

    self.assertEqual(mlp.scope_name, "scope/" + unique_name)
    self.assertEqual(mlp.module_name, unique_name)

  @parameterized.NamedParameters(
      ("MLPNoFinalActBias", False, True),
      ("MLPNoFinalActNoBias", False, False),
      ("MLPFinalActBias", True, True),
      ("MLPFinalActNoBias", True, False),
  )
  def testConstructor(self, activate_final, use_bias):
    with self.assertRaisesRegexp(ValueError, "output_sizes must not be empty"):
      mlp = snt.nets.MLP(name=self.module_name,
                         output_sizes=[],
                         activate_final=activate_final,
                         use_bias=use_bias)

    with self.assertRaisesRegexp(KeyError, "Invalid initializer keys.*"):
      mlp = snt.nets.MLP(
          name=self.module_name,
          output_sizes=self.output_sizes,
          initializers={"not_w": tf.truncated_normal_initializer(stddev=1.0)},
          activate_final=activate_final,
          use_bias=use_bias)

    with self.assertRaisesRegexp(TypeError,
                                 "Initializer for 'w' is not a callable "
                                 "function or dictionary"):
      mlp = snt.nets.MLP(name=self.module_name,
                         output_sizes=self.output_sizes,
                         initializers={"w": tf.zeros([1, 2, 3])},
                         activate_final=activate_final,
                         use_bias=use_bias)

    with self.assertRaisesRegexp(TypeError,
                                 "Input 'activation' must be callable"):
      mlp = snt.nets.MLP(name=self.module_name,
                         output_sizes=self.output_sizes,
                         activation="not_a_function",
                         activate_final=activate_final,
                         use_bias=use_bias)

    with self.assertRaisesRegexp(TypeError,
                                 "output_sizes must be iterable"):
      mlp = snt.nets.MLP(name=self.module_name,
                         output_sizes=None,
                         activate_final=activate_final,
                         use_bias=use_bias)

    mlp = snt.nets.MLP(name=self.module_name,
                       output_sizes=self.output_sizes,
                       initializers=self.initializers,
                       partitioners=self.partitioners,
                       regularizers=self.regularizers,
                       activate_final=activate_final,
                       use_bias=use_bias)
    self.assertEqual(self.initializers, mlp.initializers)
    self.assertEqual(self.regularizers, mlp.regularizers)
    self.assertEqual(self.partitioners, mlp.partitioners)

    self.assertEqual(len(mlp.layers), len(self.output_sizes))
    for i in range(0, len(mlp.layers)):
      self.assertEqual(mlp.layers[i].output_size, self.output_sizes[i])

  @parameterized.NamedParameters(
      ("MLPNoFinalActBias", False, True),
      ("MLPNoFinalActNoBias", False, False),
      ("MLPFinalActBias", True, True),
      ("MLPFinalActNoBias", True, False),
  )
  def testActivateBiasFlags(self, activate_final, use_bias):
    mlp = snt.nets.MLP(name=self.module_name,
                       output_sizes=self.output_sizes,
                       activate_final=activate_final,
                       use_bias=use_bias)

    inputs = tf.placeholder(tf.float32,
                            shape=[self.batch_size, self.input_size])
    net = mlp(inputs)

    if activate_final:
      self.assertEqual(net.op.type, "Relu")
    elif use_bias:
      self.assertEqual(net.op.type, "Add")
    else:
      self.assertEqual(net.op.type, "MatMul")

    variables = mlp.get_variables()

    if use_bias:
      self.assertEqual(len(variables), len(self.output_sizes) * 2)
    else:
      self.assertEqual(len(variables), len(self.output_sizes))

  def testShape(self):
    inputs = tf.placeholder(tf.float32,
                            shape=[self.batch_size, self.input_size])
    mlp = snt.nets.MLP(name=self.module_name, output_sizes=self.output_sizes)
    output = mlp(inputs)
    self.assertTrue(output.get_shape().is_compatible_with(
        [self.batch_size, self.output_sizes[-1]]))
    self.assertEqual((self.batch_size, self.input_size), mlp.input_shape)
    self.assertEqual(self.output_sizes, list(mlp.output_sizes))

  @parameterized.NamedParameters(
      ("MLPNoFinalActBias", False, True),
      ("MLPNoFinalActNoBias", False, False),
      ("MLPFinalActBias", True, True),
      ("MLPFinalActNoBias", True, False),
  )
  def testRegularizersInRegularizationLosses(self, active_final, use_bias):
    if use_bias:
      regularizers = {"w": tf.contrib.layers.l1_regularizer(scale=0.5),
                      "b": tf.contrib.layers.l2_regularizer(scale=0.5)}
    else:
      regularizers = {"w": tf.contrib.layers.l1_regularizer(scale=0.5)}

    inputs = tf.placeholder(tf.float32,
                            shape=[self.batch_size, self.input_size])
    mlp = snt.nets.MLP(name=self.module_name, output_sizes=self.output_sizes,
                       regularizers=regularizers)
    mlp(inputs)

    graph_regularizers = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    self.assertRegexpMatches(graph_regularizers[0].name, ".*l1_regularizer.*")
    if use_bias:
      self.assertRegexpMatches(graph_regularizers[1].name, ".*l2_regularizer.*")

  @parameterized.NamedParameters(
      ("MLPNoFinalActBias", False, True),
      ("MLPNoFinalActNoBias", False, False),
      ("MLPFinalActBias", True, True),
      ("MLPFinalActNoBias", True, False),
  )
  def testTranspose(self, activate_final, use_bias):
    with tf.variable_scope("scope1"):
      mlp = snt.nets.MLP(name=self.module_name,
                         output_sizes=self.output_sizes,
                         activate_final=activate_final,
                         use_bias=use_bias)
    with tf.variable_scope("scope2"):
      mlp_transpose = mlp.transpose()

    self.assertEqual("scope1/" + self.module_name, mlp.scope_name)
    self.assertEqual(self.module_name, mlp.module_name)
    self.assertEqual("scope2/" + self.module_name + "_transpose",
                     mlp_transpose.scope_name)
    self.assertEqual(self.module_name + "_transpose",
                     mlp_transpose.module_name)

    input_to_mlp = tf.placeholder(tf.float32,
                                  shape=[self.batch_size, self.input_size])

    with self.assertRaisesRegexp(snt.Error,
                                 "Variables in {} not instantiated yet, "
                                 "__call__ the module first."
                                 .format(mlp.layers[-1].scope_name)):
      mlp_transpose(input_to_mlp)

    mlp_transpose = mlp.transpose(name="another_mlp_transpose")
    mlp_out = mlp(input_to_mlp)
    mlp_transposed_output = mlp_transpose(mlp_out)

    self.assertEqual(mlp_transposed_output.get_shape(),
                     input_to_mlp.get_shape())
    self.assertEqual(mlp_transpose.use_bias, mlp.use_bias)
    self.assertEqual(mlp_transpose.activate_final, mlp.activate_final)

    if activate_final:
      self.assertEqual(mlp_transposed_output.op.type, "Relu")
    elif use_bias:
      self.assertEqual(mlp_transposed_output.op.type, "Add")
    else:
      self.assertEqual(mlp_transposed_output.op.type, "MatMul")

    for i in range(0, len(mlp.layers)):
      self.assertEqual(mlp_transpose.layers[i].output_size,
                       mlp.layers[-1 - i].input_shape[1])

    data = np.random.rand(self.batch_size, self.input_size)
    init = tf.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(init)
      sess.run(mlp_transposed_output, feed_dict={input_to_mlp: data})

    variables = mlp_transpose.get_variables()

    if use_bias:
      self.assertEqual(len(variables), len(self.output_sizes) * 2)
    else:
      self.assertEqual(len(variables), len(self.output_sizes))

    # Test transpose method's activate_final arg.
    mlp_activate_final = mlp.transpose(activate_final=True)
    mlp_no_activate_final = mlp.transpose(activate_final=False)
    mlp_inherit_activate_final = mlp.transpose()
    self.assertEqual(True, mlp_activate_final.activate_final)
    self.assertEqual(False, mlp_no_activate_final.activate_final)
    self.assertEqual(mlp.activate_final,
                     mlp_inherit_activate_final.activate_final)

  def testVariableMap(self):
    """Tests for regressions in variable names."""

    use_bias = True
    var_names_w = [
        u"mlp/linear_0/w:0",
        u"mlp/linear_1/w:0",
        u"mlp/linear_2/w:0",
    ]
    var_names_b = [
        u"mlp/linear_0/b:0",
        u"mlp/linear_1/b:0",
        u"mlp/linear_2/b:0",
    ]
    correct_variable_names = set(var_names_w + var_names_b)

    mlp = snt.nets.MLP(name=self.module_name,
                       output_sizes=self.output_sizes,
                       activate_final=False,
                       use_bias=use_bias)

    input_shape = [10, 100]
    input_to_net = tf.placeholder(tf.float32, shape=input_shape)

    _ = mlp(input_to_net)

    variable_names = [var.name for var in mlp.get_variables()]

    self.assertEqual(set(variable_names), set(correct_variable_names))

  def testCustomGettersUsed(self):
    pi = 3.1415

    def get_pi(getter, *args, **kwargs):
      """A custom getter which sets all variables to pi."""
      variable = getter(*args, **kwargs)
      return variable * 0.0 + pi

    mlpi = snt.nets.MLP(output_sizes=[10], custom_getter=get_pi)
    mlpi(tf.zeros(shape=(2, 1)))
    mlp_variables = [mlpi.layers[0].w, mlpi.layers[0].b]

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      for var_value in sess.run(mlp_variables):
        self.assertAllClose(var_value, np.zeros_like(var_value)+pi)


if __name__ == "__main__":
  tf.test.main()
