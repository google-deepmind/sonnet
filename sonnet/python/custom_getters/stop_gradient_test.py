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
"""Tests for sonnet.python.modules.custom_getters.stop_gradient."""

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


class StopGradientTest(parameterized.ParameterizedTestCase, tf.test.TestCase):

  def testUsage(self):
    with tf.variable_scope("", custom_getter=snt.custom_getters.stop_gradient):
      lin1 = snt.Linear(10, name="linear1")

    x = tf.placeholder(tf.float32, [10, 10])
    y = lin1(x)

    variables = tf.trainable_variables()
    variable_names = [v.name for v in variables]

    self.assertEqual(2, len(variables))

    self.assertIn("linear1/w:0", variable_names)
    self.assertIn("linear1/b:0", variable_names)

    grads = tf.gradients(y, variables)

    names_to_grads = {var.name: grad for var, grad in zip(variables, grads)}

    self.assertEqual(None, names_to_grads["linear1/w:0"])
    self.assertEqual(None, names_to_grads["linear1/b:0"])

  @parameterized.NamedParameters(
      ("StopIdentity", snt.custom_getters.stop_gradient, _identity_getter),
      ("IdentityStop", _identity_getter, snt.custom_getters.stop_gradient),
  )
  def testNest(self, getter1, getter2):
    with tf.variable_scope("scope1", custom_getter=getter1):
      with tf.variable_scope("scope2", custom_getter=getter2):
        w = tf.get_variable("w", [10, 10], tf.float32)

    grads = tf.gradients(w, tf.global_variables())

    self.assertEqual(grads, [None])

  @parameterized.NamedParameters(
      ("ConvNet2D", snt.nets.ConvNet2D, _CONV_NET_2D_KWARGS, [1, 13, 13, 3]),
      ("MLP", snt.nets.MLP, _MLP_KWARGS, [1, 16]),
  )
  def testComplex(self, module, kwargs, input_shape):
    with tf.variable_scope("", custom_getter=snt.custom_getters.stop_gradient):
      conv_net = module(**kwargs)

    x = tf.placeholder(tf.float32, input_shape)
    y = conv_net(x)

    variables = tf.global_variables()
    grads = tf.gradients(y, variables)

    self.assertEqual(grads, [None] * len(variables))


if __name__ == "__main__":
  tf.test.main()
