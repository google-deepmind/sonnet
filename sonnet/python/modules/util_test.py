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

"""Tests for sonnet.python.modules.util."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

# Dependency imports
import numpy as np
import sonnet as snt
import sonnet.python.modules.util as util
import tensorflow as tf

_EXPECTED_FORMATTED_VARIABLE_LIST = (
    "Variable  Shape  Type     Collections                            Device\n"
    "m1/v1:0   3x4    float32  global_variables, trainable_variables\n"
    "m2/v2:0   5x6    float32  local_variables                        "
    "/device:GPU:*"
)

_EXPECTED_FORMATTED_VARIABLE_MAP = (
    "Key  Variable  Shape  Type     Collections                            "
    "Device\n"
    "vv1  m1/v1:0   3x4    float32  global_variables, trainable_variables\n"
    "vv2  m2/v2:0   5x6    float32  local_variables                        "
    "/device:GPU:*"
)


class UtilTest(tf.test.TestCase):

  def testQueryInModule(self):
    module = snt.Linear(output_size=42, name="linear")

    with self.assertRaisesRegexp(snt.Error, "not instantiated yet"):
      module.get_variables()

    # Compare to the desired result set, after connection.
    input_ = tf.placeholder(tf.float32, shape=[3, 4])
    _ = module(input_)
    self.assertEqual(set(module.get_variables()),
                     {module.w, module.b})
    self.assertEqual(set(snt.get_variables_in_module(module)),
                     {module.w, module.b})

  def testScopeQuery(self):
    with tf.variable_scope("prefix") as s1:
      v1 = tf.get_variable("a", shape=[3, 4])
    with tf.variable_scope("prefix_with_more_stuff") as s2:
      v2 = tf.get_variable("b", shape=[5, 6])
      v3 = tf.get_variable("c", shape=[7])

    # get_variables_in_scope should add a "/" to only search that scope, not
    # any others which share the same prefix.
    self.assertEqual(snt.get_variables_in_scope(s1), (v1,))
    self.assertEqual(set(snt.get_variables_in_scope(s2)), {v2, v3})
    self.assertEqual(snt.get_variables_in_scope(s1.name), (v1,))
    self.assertEqual(set(snt.get_variables_in_scope(s2.name)), {v2, v3})

  def testIsScopePrefix(self):
    self.assertTrue(util._is_scope_prefix("a/b/c", ""))
    self.assertTrue(util._is_scope_prefix("a/b/c", "a/b/c"))
    self.assertTrue(util._is_scope_prefix("a/b/c", "a/b"))
    self.assertTrue(util._is_scope_prefix("a/b/c", "a"))
    self.assertTrue(util._is_scope_prefix("a/b/c", "a/"))
    self.assertFalse(util._is_scope_prefix("a/b/c", "b"))
    self.assertFalse(util._is_scope_prefix("ab/c", "a"))

  def testGetNormalizedVariableMapScope(self):
    with tf.variable_scope("prefix") as s1:
      v1 = tf.get_variable("a", shape=[5, 6])
      v2 = tf.get_variable("b", shape=[7])

    variable_map = snt.get_normalized_variable_map(s1)

    self.assertEqual(len(variable_map), 2)
    self.assertIn("a:0", variable_map)
    self.assertIn("b:0", variable_map)
    self.assertIs(variable_map["a:0"], v1)
    self.assertIs(variable_map["b:0"], v2)

  def testGetNormalizedVariableMapScopeContext(self):
    with tf.variable_scope("prefix1") as s1:
      with tf.variable_scope("prefix2") as s2:
        v1 = tf.get_variable("a", shape=[5, 6])
        v2 = tf.get_variable("b", shape=[7])

    with tf.variable_scope("prefix") as s3:
      _ = tf.get_variable("c", shape=[8])

    err = r"Scope 'prefix1/prefix2' is not prefixed by 'prefix'."
    with self.assertRaisesRegexp(ValueError, err):
      variable_map = snt.get_normalized_variable_map(s2, context=s3)

    variable_map = snt.get_normalized_variable_map(s2, context=s1)
    self.assertEqual(snt.get_normalized_variable_map(s2.name, context=s1),
                     variable_map)
    self.assertEqual(snt.get_normalized_variable_map(s2.name, context=s1.name),
                     variable_map)

    self.assertEqual(len(variable_map), 2)
    self.assertIn("prefix2/a:0", variable_map)
    self.assertIn("prefix2/b:0", variable_map)
    self.assertIs(variable_map["prefix2/a:0"], v1)
    self.assertIs(variable_map["prefix2/b:0"], v2)

    with tf.variable_scope("") as s4:
      self.assertEqual(s4.name, "")

    variable_map = snt.get_normalized_variable_map(s2, context=s4)
    self.assertEqual(snt.get_normalized_variable_map(s2.name, context=s4),
                     variable_map)
    self.assertEqual(snt.get_normalized_variable_map(s2.name, context=s4.name),
                     variable_map)

    self.assertEqual(len(variable_map), 2)
    self.assertIn("prefix1/prefix2/a:0", variable_map)
    self.assertIn("prefix1/prefix2/b:0", variable_map)
    self.assertIs(variable_map["prefix1/prefix2/a:0"], v1)
    self.assertIs(variable_map["prefix1/prefix2/b:0"], v2)

  def testGetNormalizedVariableMapModule(self):
    input_ = tf.placeholder(tf.float32, shape=[1, 10, 10, 3])
    conv = snt.Conv2D(output_channels=3, kernel_shape=3)
    conv(input_)

    variable_map = snt.get_normalized_variable_map(conv)

    self.assertEqual(len(variable_map), 2)
    self.assertIn("w:0", variable_map)
    self.assertIn("b:0", variable_map)
    self.assertIs(variable_map["w:0"], conv.w)
    self.assertIs(variable_map["b:0"], conv.b)

  def testGetNormalizedVariableMapWithPartitionedVariable(self):
    hidden = tf.ones(shape=(1, 16, 16, 3))
    partitioner = tf.variable_axis_size_partitioner(4)
    conv = snt.Conv2D(output_channels=3,
                      kernel_shape=3,
                      stride=1,
                      partitioners={"w": partitioner})
    conv(hidden)
    variable_map = snt.get_normalized_variable_map(conv,
                                                   group_sliced_variables=True)
    self.assertEqual(len(variable_map), 2)
    self.assertEqual(variable_map["b:0"], conv.b)
    self.assertEqual(len(variable_map["w"]), 3)

    variable_map = snt.get_normalized_variable_map(conv,
                                                   group_sliced_variables=False)
    self.assertEqual(variable_map["b:0"], conv.b)
    self.assertEqual(set(variable_map), set(["b:0", "w/part_0:0", "w/part_1:0",
                                             "w/part_2:0"]))

  def testVariableMapItems(self):
    hidden = tf.ones(shape=(1, 16, 16, 3))
    partitioner = tf.variable_axis_size_partitioner(4)
    conv = snt.Conv2D(output_channels=3,
                      kernel_shape=3,
                      stride=1,
                      partitioners={"w": partitioner})
    conv(hidden)
    variable_map = snt.get_normalized_variable_map(conv,
                                                   group_sliced_variables=True)
    items = snt.variable_map_items(variable_map)

    items_str = sorted((key, var.name) for key, var in items)
    self.assertEqual(
        items_str,
        [(u"b:0", u"conv_2d/b:0"), ("w", u"conv_2d/w/part_0:0"),
         ("w", u"conv_2d/w/part_1:0"), ("w", u"conv_2d/w/part_2:0")])

  def testGetSaverScope(self):
    with tf.variable_scope("prefix") as s1:
      tf.get_variable("a", shape=[5, 6])
      tf.get_variable("b", shape=[7])

    saver = snt.get_saver(s1)
    self.assertIsInstance(saver, tf.train.Saver)
    self.assertIn("a:0", saver._var_list)
    self.assertIn("b:0", saver._var_list)

  def testGetSaverModule(self):
    input_ = tf.placeholder(tf.float32, shape=[1, 10, 10, 3])
    conv = snt.Conv2D(output_channels=3, kernel_shape=3)
    conv(input_)
    saver = snt.get_saver(conv)
    self.assertIsInstance(saver, tf.train.Saver)
    self.assertIn("w:0", saver._var_list)
    self.assertIn("b:0", saver._var_list)

  def testCollectionGetVariableInScope(self):
    with tf.variable_scope("prefix") as s1:
      tf.get_variable("a", shape=[1], collections=["test"], trainable=False)

    self.assertEqual(len(snt.get_variables_in_scope(s1)), 0)
    self.assertEqual(len(snt.get_variables_in_scope(s1, collection="test2")), 0)
    self.assertEqual(len(snt.get_variables_in_scope(s1, collection="test")), 1)

  def testCollectionGetSaver(self):
    with tf.variable_scope("prefix") as s1:
      input_ = tf.placeholder(tf.float32, shape=[3, 4])
      net = snt.Linear(10)(input_)
      net = snt.BatchNorm()(net, is_training=True)

    saver1 = snt.get_saver(s1)
    saver2 = snt.get_saver(s1, collections=(tf.GraphKeys.TRAINABLE_VARIABLES,))

    self.assertIsInstance(saver1, tf.train.Saver)
    self.assertIsInstance(saver2, tf.train.Saver)

    self.assertEqual(len(saver1._var_list), 5)
    self.assertIn("linear/w:0", saver1._var_list)
    self.assertIn("linear/b:0", saver1._var_list)
    self.assertIn("batch_norm/beta:0", saver1._var_list)
    self.assertIn("batch_norm/moving_mean:0", saver1._var_list)
    self.assertIn("batch_norm/moving_variance:0", saver1._var_list)

    self.assertEqual(len(saver2._var_list), 3)
    self.assertIn("linear/w:0", saver2._var_list)
    self.assertIn("linear/b:0", saver2._var_list)
    self.assertIn("batch_norm/beta:0", saver2._var_list)
    self.assertNotIn("batch_norm/moving_mean:0", saver2._var_list)
    self.assertNotIn("batch_norm/moving_variance:0", saver2._var_list)

  def testCheckInitializers(self):
    initializers = {"key_a": tf.truncated_normal_initializer(mean=0,
                                                             stddev=1),
                    "key_c": tf.truncated_normal_initializer(mean=0,
                                                             stddev=1)}
    keys = ["key_a", "key_b"]
    self.assertRaisesRegexp(KeyError,
                            "Invalid initializer keys.*",
                            snt.check_initializers,
                            initializers=initializers,
                            keys=keys)

    del initializers["key_c"]
    initializers["key_b"] = "not a function"
    self.assertRaisesRegexp(TypeError,
                            "Initializer for.*",
                            snt.check_initializers,
                            initializers=initializers,
                            keys=keys)

    initializers["key_b"] = {"key_c": "not a function"}
    self.assertRaisesRegexp(TypeError,
                            "Initializer for.*",
                            snt.check_initializers,
                            initializers=initializers,
                            keys=keys)

    initializers["key_b"] = {"key_c": tf.truncated_normal_initializer(mean=0,
                                                                      stddev=1),
                             "key_d": tf.truncated_normal_initializer(mean=0,
                                                                      stddev=1)}
    snt.check_initializers(initializers=initializers, keys=keys)

  def testCheckPartitioners(self):
    partitioners = {"key_a": tf.variable_axis_size_partitioner(10),
                    "key_c": tf.variable_axis_size_partitioner(10)}
    keys = ["key_a", "key_b"]
    self.assertRaisesRegexp(KeyError,
                            "Invalid partitioner keys.*",
                            snt.check_partitioners,
                            partitioners=partitioners,
                            keys=keys)

    del partitioners["key_c"]
    partitioners["key_b"] = "not a function"
    self.assertRaisesRegexp(TypeError,
                            "Partitioner for.*",
                            snt.check_partitioners,
                            partitioners=partitioners,
                            keys=keys)

    partitioners["key_b"] = {"key_c": "not a function"}
    self.assertRaisesRegexp(TypeError,
                            "Partitioner for.*",
                            snt.check_partitioners,
                            partitioners=partitioners,
                            keys=keys)

    partitioners["key_b"] = {"key_c": tf.variable_axis_size_partitioner(10),
                             "key_d": tf.variable_axis_size_partitioner(10)}
    snt.check_partitioners(partitioners=partitioners, keys=keys)

  def testCheckRegularizers(self):
    regularizers = {"key_a": tf.contrib.layers.l1_regularizer(scale=0.5),
                    "key_c": tf.contrib.layers.l2_regularizer(scale=0.5)}
    keys = ["key_a", "key_b"]
    self.assertRaisesRegexp(KeyError,
                            "Invalid regularizer keys.*",
                            snt.check_regularizers,
                            regularizers=regularizers,
                            keys=keys)

    del regularizers["key_c"]
    regularizers["key_b"] = "not a function"
    self.assertRaisesRegexp(TypeError,
                            "Regularizer for.*",
                            snt.check_regularizers,
                            regularizers=regularizers,
                            keys=keys)

    regularizers["key_b"] = {"key_c": "not a function"}
    self.assertRaisesRegexp(TypeError,
                            "Regularizer for.*",
                            snt.check_regularizers,
                            regularizers=regularizers,
                            keys=keys)

    regularizers["key_b"] = {
        "key_c": tf.contrib.layers.l1_regularizer(scale=0.5),
        "key_d": tf.contrib.layers.l2_regularizer(scale=0.5)}
    snt.check_regularizers(regularizers=regularizers, keys=keys)

  def testHasVariableScope(self):
    self.assertFalse(snt.has_variable_scope("string"))
    linear = snt.Linear(10)
    self.assertTrue(snt.has_variable_scope(linear))
    linear(tf.ones((10, 10)))
    self.assertTrue(snt.has_variable_scope(linear))

  def testFormatVariables(self):
    with tf.variable_scope("m1"):
      v1 = tf.get_variable("v1", shape=[3, 4])
    with tf.device("/gpu"):
      with tf.variable_scope("m2"):
        v2 = tf.get_local_variable("v2", shape=[5, 6])
    self.assertEqual(snt.format_variables([v2, v1]),
                     _EXPECTED_FORMATTED_VARIABLE_LIST)

  def testFormatVariableMap(self):
    with tf.variable_scope("m1"):
      v1 = tf.get_variable("v1", shape=[3, 4])
    with tf.device("/gpu"):
      with tf.variable_scope("m2"):
        v2 = tf.get_local_variable("v2", shape=[5, 6])
    var_map = {"vv1": v1, "vv2": v2}
    self.assertEqual(snt.format_variable_map(var_map),
                     _EXPECTED_FORMATTED_VARIABLE_MAP)

  def testLogVariables(self):
    tf.get_default_graph().add_to_collection("config", {"version": 1})
    with tf.variable_scope("m1"):
      tf.get_variable("v1", shape=[3, 4])
    with tf.device("/gpu"):
      with tf.variable_scope("m2"):
        tf.get_local_variable("v2", shape=[5, 6])
    snt.log_variables()

  def testLogVariables_with_arg(self):
    tf.get_default_graph().add_to_collection("config", {"version": 1})
    with tf.variable_scope("m1"):
      v1 = tf.get_variable("v1", shape=[3, 4])
    with tf.device("/gpu"):
      with tf.variable_scope("m2"):
        v2 = tf.get_local_variable("v2", shape=[5, 6])
    snt.log_variables([v2, v1])


class ReuseVarsTest(tf.test.TestCase):

  class VariableContainer(object):

    def __init__(self, name):
      with tf.variable_scope(name) as vs:
        self.variable_scope = vs

    @util.reuse_variables
    def method_with_reuse(self):
      return tf.get_variable("a", shape=[1])

    def method_without_reuse(self):
      return tf.get_variable("b", shape=[1])

  class InheritedVariableContainer(VariableContainer):

    @util.reuse_variables
    def not_inherited_method_with_reuse(self):
      return tf.get_variable("c", shape=[1])

  def test_reuse_method(self):
    obj1 = ReuseVarsTest.VariableContainer("scope1")
    obj2 = ReuseVarsTest.VariableContainer("scope2")

    self.assertEqual("b:0", obj1.method_without_reuse().name)
    self.assertRaisesRegexp(ValueError,
                            r"Variable b already exists, disallowed.*",
                            obj1.method_without_reuse)
    self.assertRaisesRegexp(ValueError,
                            r"Variable b already exists, disallowed.*",
                            obj2.method_without_reuse)

    self.assertEqual("scope1/a:0",
                     obj1.method_with_reuse().name)
    self.assertEqual("scope1/a:0",
                     obj1.method_with_reuse().name)

    self.assertEqual("scope2/a:0",
                     obj2.method_with_reuse().name)
    self.assertEqual("scope2/a:0",
                     obj2.method_with_reuse().name)

  def test_multiple_objects_per_variable_scope(self):
    obj1 = ReuseVarsTest.VariableContainer("scope1")
    obj2 = ReuseVarsTest.VariableContainer("scope1")

    self.assertEqual("scope1/a:0",
                     obj1.method_with_reuse().name)
    self.assertEqual("scope1/a:0",
                     obj1.method_with_reuse().name)

    self.assertEqual("scope1/a:0",
                     obj2.method_with_reuse().name)
    self.assertEqual("scope1/a:0",
                     obj2.method_with_reuse().name)

  def test_reuse_inherited_method(self):
    obj1 = ReuseVarsTest.InheritedVariableContainer("scope1")
    obj2 = ReuseVarsTest.InheritedVariableContainer("scope2")

    self.assertEqual("b:0", obj1.method_without_reuse().name)
    self.assertRaisesRegexp(ValueError,
                            r"Variable b already exists, disallowed.*",
                            obj1.method_without_reuse)
    self.assertRaisesRegexp(ValueError,
                            r"Variable b already exists, disallowed.*",
                            obj2.method_without_reuse)

    self.assertEqual("scope1/a:0", obj1.method_with_reuse().name)
    self.assertEqual("scope1/a:0", obj1.method_with_reuse().name)
    self.assertEqual("scope1/c:0", obj1.not_inherited_method_with_reuse().name)
    self.assertEqual("scope1/c:0", obj1.not_inherited_method_with_reuse().name)

    self.assertEqual("scope2/a:0", obj2.method_with_reuse().name)
    self.assertEqual("scope2/a:0", obj2.method_with_reuse().name)
    self.assertEqual("scope2/c:0", obj2.not_inherited_method_with_reuse().name)
    self.assertEqual("scope2/c:0", obj2.not_inherited_method_with_reuse().name)

  def test_reuse_abstract_module(self):

    class ModuleReuse(snt.AbstractModule):

      def __init__(self, shape, name="multi_template_test"):
        super(ModuleReuse, self).__init__(name=name)
        self._shape = shape

      @util.reuse_variables
      def a(self):
        return tf.get_variable("a", shape=self._shape)

      @util.reuse_variables
      def add_b(self, inputs):
        return inputs + tf.get_variable("b", shape=self._shape)

      def _build(self, inputs):
        return self.add_b(inputs + self.a())

    np.random.seed(100)
    batch_size = 3
    in_size = 4
    inputs = tf.placeholder(tf.float32, shape=[batch_size, in_size])

    module1 = ModuleReuse(inputs.get_shape().as_list())
    module2 = ModuleReuse(inputs.get_shape().as_list())

    a1 = module1.a()
    inputs_plus_b1 = module1.add_b(inputs)
    inputs_plus_ab1 = module1(inputs)  # pylint: disable=not-callable

    inputs_plus_ab2 = module2(inputs)  # pylint: disable=not-callable
    inputs_plus_b2 = module2.add_b(inputs)
    a2 = module2.a()

    inputs_plus_ab1_again = module1(inputs)  # pylint: disable=not-callable
    inputs_plus_ab2_again = module2(inputs)  # pylint: disable=not-callable

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      input_data = np.random.rand(batch_size, in_size)
      out = sess.run([a1, inputs_plus_b1, inputs_plus_ab1, a2, inputs_plus_b2,
                      inputs_plus_ab2],
                     feed_dict={inputs: input_data})

      self.assertNotAlmostEqual(np.linalg.norm(out[0] - out[3]), 0)
      self.assertNotAlmostEqual(np.linalg.norm(out[1] - out[4]), 0)
      self.assertNotAlmostEqual(np.linalg.norm(out[2] - out[5]), 0)

      self.assertAllClose(out[0] + out[1], out[2])
      self.assertAllClose(out[3] + out[4], out[5])

      out = sess.run([inputs_plus_ab1, inputs_plus_ab1_again],
                     feed_dict={inputs: input_data})
      self.assertAllEqual(out[0], out[1])

      out = sess.run([inputs_plus_ab2, inputs_plus_ab2_again],
                     feed_dict={inputs: input_data})
      self.assertAllEqual(out[0], out[1])

  def test_variable_scope_call_order(self):
    class TestModule(snt.AbstractModule):

      def __init__(self, name="test_module"):
        super(TestModule, self).__init__(name=name)

      @util.reuse_variables
      def a(self):
        return self.scope_name

      def _build(self):
        pass

      @property
      def variable_scope(self):
        # Needed to access `self.variable_scope` before calling `self.build()`.
        return self._template.variable_scope

    m1 = TestModule(name="m1")
    m2 = TestModule(name="m2")

    a1 = m1.a
    a2 = m2.a

    self.assertEqual("m1", a1())
    self.assertEqual("m2", a2())

  def test_multiple_graphs(self):
    g1 = tf.Graph()
    g2 = tf.Graph()

    with g1.as_default():
      obj1 = ReuseVarsTest.VariableContainer("scope1")
      obj2 = ReuseVarsTest.VariableContainer("scope1")

      self.assertEqual("scope1/a:0",
                       obj1.method_with_reuse().name)
      self.assertEqual("scope1/a:0",
                       obj1.method_with_reuse().name)

      self.assertEqual("scope1/a:0",
                       obj2.method_with_reuse().name)
      self.assertEqual("scope1/a:0",
                       obj2.method_with_reuse().name)

    with g2.as_default():
      obj1 = ReuseVarsTest.VariableContainer("scope1")
      obj2 = ReuseVarsTest.VariableContainer("scope1")

      self.assertEqual("scope1/a:0",
                       obj1.method_with_reuse().name)
      self.assertEqual("scope1/a:0",
                       obj1.method_with_reuse().name)

      self.assertEqual("scope1/a:0",
                       obj2.method_with_reuse().name)
      self.assertEqual("scope1/a:0",
                       obj2.method_with_reuse().name)

  def test_name_scopes(self):

    class VariableContainerWithOps(ReuseVarsTest.VariableContainer):

      @util.reuse_variables
      def add_b(self, tensor):
        b = tf.get_variable("b", shape=[1])
        return tensor + b

      @util.reuse_variables
      def add_a(self, tensor):
        return tensor + self.method_with_reuse()

      @util.reuse_variables
      def nested_add(self, tensor):
        return tf.ones(shape=[1]) + self.add_a(tensor)

    def get_tensor_names_from_default_graph():
      ops = [
          op for op in tf.get_default_graph().get_operations()
          if "Initializer" not in op.name and "Assign" not in op.name and
          "read" not in op.name
      ]
      tensor_names = []
      for op in ops:
        tensor_names.extend(tensor.name for tensor in op.outputs)
      return tensor_names

    obj1 = VariableContainerWithOps("scope1")
    obj2 = VariableContainerWithOps("scope2")
    zeros = tf.zeros(shape=[1])

    self.assertEqual("scope1/add_b/add:0", obj1.add_b(zeros).name)
    self.assertEqual("scope1/add_b_1/add:0", obj1.add_b(zeros).name)

    self.assertEqual("scope1/add_a/add:0", obj1.add_a(zeros).name)
    self.assertEqual("scope1/add_a_1/add:0", obj1.add_a(zeros).name)

    self.assertEqual("scope1/nested_add/add:0",
                     obj1.nested_add(zeros).name)
    self.assertEqual("scope1/nested_add_1/add:0",
                     obj1.nested_add(zeros).name)

    ones = tf.ones(shape=[1])
    self.assertEqual("scope2/add_b/add:0", obj2.add_b(ones).name)
    self.assertEqual("scope2/add_b_1/add:0", obj2.add_b(ones).name)

    self.assertEqual("scope2/add_a/add:0", obj2.add_a(ones).name)
    self.assertEqual("scope2/add_a_1/add:0", obj2.add_a(ones).name)

    self.assertEqual("scope2/nested_add/add:0",
                     obj2.nested_add(ones).name)
    self.assertEqual("scope2/nested_add_1/add:0",
                     obj2.nested_add(ones).name)

    tensor_names = [
        "zeros:0",
        "scope1/b:0",
        "scope1/add_b/add:0",
        "scope1/add_b_1/add:0",
        "scope1/a:0",
        "scope1/add_a/add:0",
        "scope1/add_a_1/add:0",
        "scope1/nested_add/ones:0",
        "scope1/add_a_2/add:0",
        "scope1/nested_add/add:0",
        "scope1/nested_add_1/ones:0",
        "scope1/add_a_3/add:0",
        "scope1/nested_add_1/add:0",
        "ones:0",
        "scope2/b:0",
        "scope2/add_b/add:0",
        "scope2/add_b_1/add:0",
        "scope2/a:0",
        "scope2/add_a/add:0",
        "scope2/add_a_1/add:0",
        "scope2/nested_add/ones:0",
        "scope2/add_a_2/add:0",
        "scope2/nested_add/add:0",
        "scope2/nested_add_1/ones:0",
        "scope2/add_a_3/add:0",
        "scope2/nested_add_1/add:0",
    ]

    self.assertEqual(tensor_names, get_tensor_names_from_default_graph())


class NameFunctionTest(tf.test.TestCase):

  def testToSnakeCase(self):
    test_cases = [
        ("UpperCamelCase", "upper_camel_case"),
        ("lowerCamelCase", "lower_camel_case"),
        ("endsWithXYZ", "ends_with_xyz"),
        ("already_snake_case", "already_snake_case"),
        ("__private__", "private"),
        ("LSTMModule", "lstm_module"),
        ("version123p56vfxObject", "version_123p56vfx_object"),
        ("version123P56VFXObject", "version_123p56vfx_object"),
        ("versionVFX123P56Object", "version_vfx123p56_object"),
        ("versionVfx123P56Object", "version_vfx_123p56_object"),
        ("lstm1", "lstm_1"),
        ("LSTM1", "lstm1"),
    ]
    for camel_case, snake_case in test_cases:
      actual = util.to_snake_case(camel_case)
      self.assertEqual(actual, snake_case, "_to_snake_case(%s) -> %s != %s" %
                       (camel_case, actual, snake_case))

  def testNameForCallable_Function(self):
    def test():
      pass

    self.assertName(test, "test")

  def testNameForCallable_Lambda(self):
    test = lambda x: x
    self.assertName(test, None)

  def testNameForCallable_Partial(self):
    def test(*unused_args):
      pass

    test = functools.partial(functools.partial(test, "a"), "b")
    self.assertName(test, "test")

  def testNameForCallable_Instance(self):
    class Test(object):

      def __call__(self):
        pass

    self.assertName(Test(), None)

  def assertName(self, func, expected):
    name = util.name_for_callable(func)
    self.assertEqual(name, expected)


if __name__ == "__main__":
  tf.test.main()
