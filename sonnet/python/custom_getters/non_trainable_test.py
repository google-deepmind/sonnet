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
"""Tests for sonnet.python.modules.custom_getters.non_trainable."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sonnet as snt
from sonnet.testing import parameterized
import tensorflow as tf

_CONV_NET_2D_KWARGS = {
    "output_channels": [16, 16],
    "kernel_shapes": [3],
    "strides": [2],
    "paddings": [snt.VALID],
}
_MLP_KWARGS = {
    "output_sizes": [16, 16],
}


def _identity_getter(getter, *args, **kwargs):
  return getter(*args, **kwargs)


class NonTrainableTest(parameterized.ParameterizedTestCase, tf.test.TestCase):

  def testUsage(self):
    with tf.variable_scope("", custom_getter=snt.custom_getters.non_trainable):
      lin1 = snt.Linear(10, name="linear1")

    x = tf.placeholder(tf.float32, [10, 10])
    lin1(x)

    self.assertEqual(2, len(tf.global_variables()))
    self.assertEqual(0, len(tf.trainable_variables()))

  @parameterized.NamedParameters(
      ("NonIdentity", snt.custom_getters.non_trainable, _identity_getter),
      ("IdentityNon", _identity_getter, snt.custom_getters.non_trainable),
  )
  def testNest(self, getter1, getter2):
    with tf.variable_scope("scope1", custom_getter=getter1):
      with tf.variable_scope("scope2", custom_getter=getter2):
        tf.get_variable("w", [10, 10], tf.float32)

    self.assertEqual(1, len(tf.global_variables()))
    self.assertEqual(0, len(tf.trainable_variables()))

  @parameterized.NamedParameters(
      ("ConvNet2D", snt.nets.ConvNet2D, _CONV_NET_2D_KWARGS, [1, 13, 13, 3]),
      ("MLP", snt.nets.MLP, _MLP_KWARGS, [1, 16]),
  )
  def testComplex(self, module, kwargs, input_shape):
    with tf.variable_scope("", custom_getter=snt.custom_getters.non_trainable):
      module_instance = module(**kwargs)

    x1 = tf.placeholder(tf.float32, input_shape)
    x2 = tf.placeholder(tf.float32, input_shape)
    module_instance(x1)
    module_instance(x2)

    self.assertNotEqual(0, len(tf.global_variables()))
    self.assertEqual(0, len(tf.trainable_variables()))


if __name__ == "__main__":
  tf.test.main()
