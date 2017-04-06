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
# =============================================================================
"""Tests for sonnet.python.modules.base."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial

import numpy as np
from sonnet.python.modules import base
import tensorflow as tf

logging = tf.logging


class ModuleWithClassKeys(base.AbstractModule):
  """Dummy module that defines some keys as class attributes."""
  POSSIBLE_INITIALIZER_KEYS = {"foo", "bar"}


class ModuleWithNoInitializerKeys(base.AbstractModule):
  """Dummy module without any intiailizer keys."""
  pass


class ModuleWithCustomInitializerKeys(base.AbstractModule):
  """Dummy module that overrides get_possible_initializer_keys."""

  @classmethod
  def get_possible_initializer_keys(cls, custom_key):
    return {"foo"} if custom_key else {"bar"}


class AbstractModuleTest(tf.test.TestCase):

  def testInitializerKeys(self):
    keys = ModuleWithClassKeys.get_possible_initializer_keys()
    self.assertEqual(keys, {"foo", "bar"})
    keys = ModuleWithNoInitializerKeys.get_possible_initializer_keys()
    self.assertEqual(keys, set())
    self.assertRaisesRegexp(
        TypeError, "takes exactly 2 arguments",
        ModuleWithCustomInitializerKeys.get_possible_initializer_keys)
    keys = ModuleWithCustomInitializerKeys.get_possible_initializer_keys(True)
    self.assertEqual(keys, {"foo"})
    keys = ModuleWithCustomInitializerKeys.get_possible_initializer_keys(False)
    self.assertEqual(keys, {"bar"})


def _make_model_with_params(inputs, output_size):
  weight_shape = [inputs.get_shape().as_list()[-1], output_size]
  weight = tf.get_variable("w", shape=weight_shape, dtype=inputs.dtype)
  return tf.matmul(inputs, weight)


class ModuleTest(tf.test.TestCase):

  def testFunctionType(self):
    with self.assertRaises(TypeError) as cm:
      base.Module(build="not_a_function")

    self.assertEqual(cm.exception.message, "Input 'build' must be callable.")

  def testSharing(self):
    batch_size = 3
    in_size = 4
    inputs1 = tf.placeholder(tf.float32, shape=[batch_size, in_size])
    inputs2 = tf.placeholder(tf.float32, shape=[batch_size, in_size])

    model = base.Module(build=partial(_make_model_with_params, output_size=10))
    outputs1 = model(inputs1)
    outputs2 = model(inputs2)
    input_data = np.random.rand(batch_size, in_size)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      outputs1, outputs2 = sess.run(
          [outputs1, outputs2],
          feed_dict={inputs1: input_data,
                     inputs2: input_data})
      self.assertAllClose(outputs1, outputs2)


if __name__ == "__main__":
  tf.test.main()
