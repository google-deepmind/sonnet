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

# Dependency imports
import sonnet as snt
import sonnet.python.modules.util as util
import tensorflow as tf


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

    self.assertEqual(len(variable_map), 2)
    self.assertIn("prefix2/a:0", variable_map)
    self.assertIn("prefix2/b:0", variable_map)
    self.assertIs(variable_map["prefix2/a:0"], v1)
    self.assertIs(variable_map["prefix2/b:0"], v2)

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
      net = snt.BatchNorm()(net)

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
    with tf.variable_scope("m2"):
      v2 = tf.get_variable("v2", shape=[5, 6])
    self.assertEquals(snt.format_variables([v2, v1]),
                      ("Variable  Shape  Type\n"
                       "m1/v1:0   3x4    tf.float32\n"
                       "m2/v2:0   5x6    tf.float32"))

  def testFormatVariableMap(self):
    with tf.variable_scope("m1"):
      v1 = tf.get_variable("v1", shape=[3, 4])
    with tf.variable_scope("m2"):
      v2 = tf.get_variable("v2", shape=[5, 6])
    var_map = {"vv1": v1, "vv2": v2}
    self.assertEquals(snt.format_variable_map(var_map),
                      ("Key  Variable  Shape  Type\n"
                       "vv1  m1/v1:0   3x4    tf.float32\n"
                       "vv2  m2/v2:0   5x6    tf.float32"))

if __name__ == "__main__":
  tf.test.main()
